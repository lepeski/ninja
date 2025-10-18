import { randomUUID } from 'crypto';
import sqlite3 from 'sqlite3';
import { ChromaClient, Collection } from 'chromadb';
import { embedSingle } from './openai.js';
import { logger } from './logger.js';

sqlite3.verbose();

export interface StoredMessage {
  id: string;
  role: 'user' | 'assistant' | 'system';
  userId: string;
  channelId: string;
  content: string;
  timestamp: number;
  embedding: number[];
}

export class MemoryStore {
  private db: sqlite3.Database;
  private shortTerm: Map<string, StoredMessage[]> = new Map();
  private collection?: Collection;
  private chromaReady = false;

  constructor(private readonly maxHistory: number) {
    this.db = new sqlite3.Database('memory.db');
  }

  async init(): Promise<void> {
    await this.run(
      `CREATE TABLE IF NOT EXISTS messages (
        id TEXT PRIMARY KEY,
        role TEXT NOT NULL,
        user_id TEXT NOT NULL,
        channel_id TEXT NOT NULL,
        content TEXT NOT NULL,
        timestamp INTEGER NOT NULL,
        embedding TEXT NOT NULL
      )`
    );

    await this.run(
      `CREATE TABLE IF NOT EXISTS user_prefs (
        user_id TEXT PRIMARY KEY,
        opted_in INTEGER NOT NULL DEFAULT 1
      )`
    );

    await this.pruneOldEntries();
    await this.initChroma();
  }

  private async initChroma() {
    try {
      const path = process.env.CHROMA_URL;
      const client = path ? new ChromaClient({ path }) : new ChromaClient();
      this.collection = await client.getOrCreateCollection({ name: 'discord-memory' });
      const rows = await this.all<{
        id: string;
        content: string;
        embedding: string;
        channel_id: string;
        user_id: string;
        role: string;
        timestamp: number;
      }>('SELECT * FROM messages');

      if (rows.length) {
        const ids: string[] = [];
        const documents: string[] = [];
        const embeddings: number[][] = [];
        const metadatas: Record<string, unknown>[] = [];

        for (const row of rows) {
          ids.push(row.id);
          documents.push(row.content);
          embeddings.push(JSON.parse(row.embedding));
          metadatas.push({
            userId: row.user_id,
            channelId: row.channel_id,
            role: row.role,
            timestamp: row.timestamp,
          });
        }

        if ('upsert' in this.collection && typeof this.collection.upsert === 'function') {
          await this.collection.upsert({ ids, documents, embeddings, metadatas });
        } else {
          await this.collection.add({ ids, documents, embeddings, metadatas });
        }
      }

      this.chromaReady = true;
      logger.info('ChromaDB collection initialised with %d items.', rows.length);
    } catch (error) {
      this.chromaReady = false;
      logger.error({ err: error }, 'Failed to initialise ChromaDB client');
    }
  }

  async rememberMessage(params: Omit<StoredMessage, 'id' | 'timestamp' | 'embedding'> & { timestamp?: number }): Promise<StoredMessage> {
    const id = randomUUID();
    const timestamp = params.timestamp ?? Date.now();
    const embedding = await embedSingle(params.content);

    const record: StoredMessage = {
      id,
      role: params.role,
      userId: params.userId,
      channelId: params.channelId,
      content: params.content,
      timestamp,
      embedding,
    };

    await this.run(
      `INSERT INTO messages (id, role, user_id, channel_id, content, timestamp, embedding)
       VALUES (?, ?, ?, ?, ?, ?, ?)`,
      [id, params.role, params.userId, params.channelId, params.content, timestamp, JSON.stringify(embedding)]
    );

    await this.addToChroma(record);
    this.pushShortTerm(record);
    return record;
  }

  private async addToChroma(message: StoredMessage) {
    if (!this.collection || !this.chromaReady) return;
    try {
      await this.collection.add({
        ids: [message.id],
        documents: [message.content],
        embeddings: [message.embedding],
        metadatas: [
          {
            userId: message.userId,
            channelId: message.channelId,
            role: message.role,
            timestamp: message.timestamp,
          },
        ],
      });
    } catch (error) {
      logger.warn({ err: error }, 'Failed to add entry to ChromaDB');
    }
  }

  private pushShortTerm(message: StoredMessage) {
    const history = this.shortTerm.get(message.channelId) ?? [];
    history.push(message);
    while (history.length > this.maxHistory) {
      history.shift();
    }
    this.shortTerm.set(message.channelId, history);
  }

  getShortTerm(channelId: string): StoredMessage[] {
    return this.shortTerm.get(channelId) ?? [];
  }

  clearShortTerm(channelId?: string) {
    if (channelId) {
      this.shortTerm.delete(channelId);
    } else {
      this.shortTerm.clear();
    }
  }

  async queryRelevantMemories(channelId: string, content: string, limit = 5): Promise<StoredMessage[]> {
    const embedding = await embedSingle(content);

    if (this.collection && this.chromaReady) {
      try {
        const response = await this.collection.query({
          queryEmbeddings: [embedding],
          nResults: limit * 2,
          where: { channelId },
        });

        const results: StoredMessage[] = [];
        const ids = response.ids?.[0] ?? [];
        const documents = response.documents?.[0] ?? [];
        const metadatas = response.metadatas?.[0] ?? [];
        const distances = response.distances?.[0] ?? [];

        ids.forEach((id, index) => {
          const distance = distances[index] ?? 1;
          const similarity = 1 - distance;
          if (similarity < 0.3) {
            return;
          }
          const metadata = metadatas[index] as Record<string, unknown>;
          results.push({
            id,
            role: (metadata.role as StoredMessage['role']) ?? 'user',
            userId: (metadata.userId as string) ?? 'unknown',
            channelId: (metadata.channelId as string) ?? channelId,
            content: documents[index] ?? '',
            timestamp: (metadata.timestamp as number) ?? Date.now(),
            embedding: [],
          });
        });

        if (results.length >= limit) {
          return results.slice(0, limit);
        }
      } catch (error) {
        logger.warn({ err: error }, 'ChromaDB query failed, falling back to SQLite search');
      }
    }

    const rows = await this.all<{
      id: string;
      role: string;
      user_id: string;
      channel_id: string;
      content: string;
      timestamp: number;
      embedding: string;
    }>(
      'SELECT * FROM messages WHERE channel_id = ? ORDER BY timestamp DESC LIMIT 100',
      [channelId]
    );

    const scored = rows
      .map((row) => {
        const pastEmbedding = JSON.parse(row.embedding) as number[];
        const similarity = cosineSimilarity(embedding, pastEmbedding);
        return {
          similarity,
          message: {
            id: row.id,
            role: row.role as StoredMessage['role'],
            userId: row.user_id,
            channelId: row.channel_id,
            content: row.content,
            timestamp: row.timestamp,
            embedding: pastEmbedding,
          } satisfies StoredMessage,
        };
      })
      .filter((item) => item.similarity >= 0.3)
      .sort((a, b) => b.similarity - a.similarity)
      .slice(0, limit)
      .map((item) => item.message);

    return scored;
  }

  async pruneOldEntries(): Promise<void> {
    const cutoff = Date.now() - 30 * 24 * 60 * 60 * 1000;
    await this.run('DELETE FROM messages WHERE timestamp < ?', [cutoff]);
  }

  async reset(): Promise<void> {
    await this.run('DELETE FROM messages');
    await this.run('DELETE FROM user_prefs');
    this.shortTerm.clear();
    if (this.collection && this.chromaReady) {
      try {
        await this.collection.delete({ where: {} });
      } catch (error) {
        logger.warn({ err: error }, 'Failed clearing ChromaDB collection');
      }
    }
  }

  async setOptIn(userId: string, optedIn: boolean) {
    await this.run(
      `INSERT INTO user_prefs (user_id, opted_in)
       VALUES (?, ?)
       ON CONFLICT(user_id) DO UPDATE SET opted_in = excluded.opted_in`,
      [userId, optedIn ? 1 : 0]
    );
  }

  async isOptedIn(userId: string): Promise<boolean> {
    const row = await this.get<{ opted_in: number }>('SELECT opted_in FROM user_prefs WHERE user_id = ?', [userId]);
    if (!row) return true;
    return row.opted_in === 1;
  }

  private run(sql: string, params: unknown[] = []): Promise<void> {
    return new Promise((resolve, reject) => {
      this.db.run(sql, params, (err) => {
        if (err) reject(err);
        else resolve();
      });
    });
  }

  private get<T = unknown>(sql: string, params: unknown[] = []): Promise<T | undefined> {
    return new Promise((resolve, reject) => {
      this.db.get(sql, params, (err, row) => {
        if (err) reject(err);
        else resolve(row as T | undefined);
      });
    });
  }

  private all<T = unknown>(sql: string, params: unknown[] = []): Promise<T[]> {
    return new Promise((resolve, reject) => {
      this.db.all(sql, params, (err, rows) => {
        if (err) reject(err);
        else resolve(rows as T[]);
      });
    });
  }
}

function cosineSimilarity(a: number[], b: number[]): number {
  if (!a.length || !b.length || a.length !== b.length) return 0;
  let dot = 0;
  let normA = 0;
  let normB = 0;
  for (let i = 0; i < a.length; i += 1) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  if (!normA || !normB) return 0;
  return dot / (Math.sqrt(normA) * Math.sqrt(normB));
}
