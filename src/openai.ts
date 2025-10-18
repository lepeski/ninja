import OpenAI from 'openai';
import { logger } from './logger.js';

const apiKey = process.env.OPENAI_API_KEY;

if (!apiKey) {
  logger.warn('OPENAI_API_KEY is not set. OpenAI features will fail.');
}

export const openai = new OpenAI({
  apiKey,
});

export interface ChatMessage {
  role: 'system' | 'user' | 'assistant';
  content: string;
}

export interface StreamChatOptions {
  messages: ChatMessage[];
  model: string;
  signal?: AbortSignal;
  onToken?: (token: string) => void | Promise<void>;
}

export async function streamChatCompletion({
  messages,
  model,
  signal,
  onToken,
}: StreamChatOptions): Promise<string> {
  const stream = await openai.responses.stream({
    model,
    messages,
  });

  if (signal) {
    signal.addEventListener('abort', () => {
      stream.abort();
    });
  }

  for await (const event of stream) {
    if (event.type === 'response.output_text.delta' && event.delta) {
      await onToken?.(event.delta);
    }
  }

  const finalText = await stream.finalText();
  return finalText ?? '';
}

export async function embedText(texts: string[]): Promise<number[][]> {
  if (!texts.length) {
    return [];
  }

  const response = await openai.embeddings.create({
    model: 'text-embedding-3-large',
    input: texts,
  });

  return response.data.map((item) => item.embedding);
}

export async function embedSingle(text: string): Promise<number[]> {
  const [embedding] = await embedText([text]);
  return embedding ?? [];
}
