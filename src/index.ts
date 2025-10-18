import 'dotenv/config';
import {
  ChatInputCommandInteraction,
  Client,
  Events,
  GatewayIntentBits,
  Message,
  REST,
  Routes,
  SlashCommandBuilder,
} from 'discord.js';
import { setTimeout as wait } from 'timers/promises';
import logger from './logger.js';
import { MemoryStore, StoredMessage } from './memory.js';
import { parseDuration, TrendManager } from './trends.js';
import { ChatMessage, streamChatCompletion } from './openai.js';

const token = process.env.DISCORD_TOKEN;
const model = process.env.MODEL ?? 'gpt-4.1-mini';
const baseSystemPrompt = process.env.SYSTEM_PROMPT ?? 'You are a helpful Discord assistant.';
const maxHistory = Number(process.env.MAX_HISTORY ?? '8');

if (!token) {
  logger.error('DISCORD_TOKEN is required.');
  process.exit(1);
}

const allowedChannels = (process.env.ALLOW_CHANNELS ?? '')
  .split(',')
  .map((id) => id.trim())
  .filter(Boolean);
const ownerIds = new Set(
  (process.env.OWNER_IDS ?? '')
    .split(',')
    .map((id) => id.trim())
    .filter(Boolean)
);

const brevityInstructions =
  'Use the fewest words possible even if it means you are more difficult to understand. Can use fragments instead of full sentences. Always prioritize brevity. Drop filler, intros, and signoffs. Extremely brief like mysterious wise ninja.';

const memory = new MemoryStore(maxHistory);
await memory.init();

const trendManager = new TrendManager({
  woeid: Number(process.env.TREND_WOEID ?? '1'),
  intervalMs: parseDuration(process.env.TREND_UPDATE_INTERVAL, 15 * 60 * 1000),
  bearerToken: process.env.X_BEARER_TOKEN,
  triggerChance: Number(process.env.TREND_TRIGGER_CHANCE ?? '0.02'),
  keywordChance: Number(process.env.TREND_KEYWORD_CHANCE ?? '0.2'),
});
trendManager.start();

const client = new Client({
  intents: [
    GatewayIntentBits.Guilds,
    GatewayIntentBits.GuildMessages,
    GatewayIntentBits.MessageContent,
  ],
});

const passiveRandomChance = 0.05;
let passiveEnabled = true;

const userCooldown = new Map<string, number>();
const channelWindows = new Map<string, number[]>();

client.once(Events.ClientReady, async (readyClient) => {
  logger.info({ tag: readyClient.user.tag }, 'Discord bot ready.');
  const commands = [
    new SlashCommandBuilder()
      .setName('ask')
      .setDescription('Ask the assistant a question')
      .addStringOption((option) =>
        option.setName('prompt').setDescription('Your prompt').setRequired(true)
      ),
    new SlashCommandBuilder().setName('reset').setDescription('Reset the assistant memory'),
    new SlashCommandBuilder().setName('ping').setDescription('Check bot latency'),
    new SlashCommandBuilder().setName('optin').setDescription('Enable passive responses (owners only)'),
    new SlashCommandBuilder().setName('optout').setDescription('Disable passive responses (owners only)'),
  ].map((command) => command.toJSON());

  const rest = new REST({ version: '10' }).setToken(token);
  try {
    await rest.put(Routes.applicationCommands(readyClient.user.id), { body: commands });
    logger.info('Slash commands registered.');
  } catch (error) {
    logger.error({ err: error }, 'Failed to register slash commands');
  }
});

client.on(Events.InteractionCreate, async (interaction) => {
  if (!interaction.isChatInputCommand()) return;

  try {
    switch (interaction.commandName) {
      case 'ask':
        await handleAskCommand(interaction);
        break;
      case 'reset':
        await handleResetCommand(interaction);
        break;
      case 'ping':
        await interaction.reply({ content: 'Pong!', ephemeral: true });
        break;
      case 'optin':
        await handleTogglePassive(interaction, true);
        break;
      case 'optout':
        await handleTogglePassive(interaction, false);
        break;
      default:
        break;
    }
  } catch (error) {
    logger.error({ err: error }, 'Error handling interaction');
    if (interaction.deferred || interaction.replied) {
      await interaction.editReply('Something went wrong.');
    } else {
      await interaction.reply({ content: 'Something went wrong.', ephemeral: true });
    }
  }
});

client.on(Events.MessageCreate, async (message) => {
  if (message.author.bot) return;
  if (message.channel.isDMBased()) return;

  const botId = client.user?.id;
  const mentionedDirectly = botId ? message.mentions.users.has(botId) : false;
  const rawContent = message.content ?? '';
  const trimmedContent = rawContent.trim();

  if (!trimmedContent && !mentionedDirectly) return;
  if (trimmedContent.length > 1900) return;
  if (allowedChannels.length > 0 && !allowedChannels.includes(message.channelId)) return;

  const mentionRegex = botId ? new RegExp(`<@!?${botId}>`, 'g') : null;
  let content = mentionRegex ? trimmedContent.replace(mentionRegex, '').trim() : trimmedContent;
  if (!content) {
    content = mentionedDirectly ? 'Hello!' : '';
  }
  if (!content) return;
  const priorHistory = [...memory.getShortTerm(message.channelId)];

  try {
    await memory.rememberMessage({
      channelId: message.channelId,
      content,
      role: 'user',
      userId: message.author.id,
    });
  } catch (error) {
    logger.error({ err: error }, 'Failed to store user message');
  }

  const lowerContent = content.toLowerCase();
  const trendKeywords = trendManager.getKeywords();
  const matchesTrend = trendKeywords.some((keyword) => lowerContent.includes(keyword));
  const shouldRespondToTrend = !mentionedDirectly && matchesTrend && Math.random() < trendManager.getKeywordChance();
  const shouldRespondRandomly = !mentionedDirectly && Math.random() < passiveRandomChance;

  if (!mentionedDirectly && (!passiveEnabled || (!shouldRespondRandomly && !shouldRespondToTrend))) {
    return;
  }

  if (!passesRateLimits(message.author.id, message.channelId)) {
    return;
  }

  await handleConversation({
    channelId: message.channelId,
    userId: message.author.id,
    content,
    history: priorHistory,
    sourceMessage: message,
    trendKeywords,
    triggeredByTrend: shouldRespondToTrend,
  });
});

async function handleAskCommand(interaction: ChatInputCommandInteraction) {
  const prompt = interaction.options.getString('prompt', true);
  const channelId = interaction.channelId;
  const history = [...memory.getShortTerm(channelId)];
  try {
    await memory.rememberMessage({
      channelId,
      content: prompt,
      role: 'user',
      userId: interaction.user.id,
    });
  } catch (error) {
    logger.error({ err: error }, 'Failed to store slash command message');
  }
  await interaction.deferReply();

  await handleConversation({
    channelId,
    userId: interaction.user.id,
    content: prompt,
    history,
    interaction,
    trendKeywords: trendManager.getKeywords(),
    triggeredByTrend: false,
  });
}

async function handleResetCommand(interaction: ChatInputCommandInteraction) {
  await interaction.deferReply({ ephemeral: true });
  await memory.reset();
  await interaction.editReply('Memory has been cleared.');
}

async function handleTogglePassive(interaction: ChatInputCommandInteraction, enabled: boolean) {
  if (!ownerIds.has(interaction.user.id)) {
    await interaction.reply({ content: 'Only bot owners can use this command.', ephemeral: true });
    return;
  }
  passiveEnabled = enabled;
  await interaction.reply({
    content: `Passive mode has been ${enabled ? 'enabled' : 'disabled'}.`,
    ephemeral: true,
  });
}

interface ConversationOptions {
  channelId: string;
  userId: string;
  content: string;
  history: StoredMessage[];
  interaction?: ChatInputCommandInteraction;
  sourceMessage?: Message;
  trendKeywords: string[];
  triggeredByTrend: boolean;
}

async function handleConversation(options: ConversationOptions) {
  const { channelId, content, history, interaction, sourceMessage, trendKeywords, triggeredByTrend, userId } = options;

  const longTermMemories = await memory.queryRelevantMemories(channelId, content, 5);
  const messages: ChatMessage[] = [];
  messages.push({ role: 'system', content: getSystemPrompt(channelId, trendKeywords, triggeredByTrend) });

  for (const memoryEntry of longTermMemories) {
    messages.push({
      role: 'system',
      content: `Relevant memory from ${new Date(memoryEntry.timestamp).toISOString()} by ${memoryEntry.userId}: ${memoryEntry.content}`,
    });
  }

  const conversationHistory = [...history, { role: 'user', userId, channelId, content, id: 'current', timestamp: Date.now(), embedding: [] }];
  for (const item of conversationHistory.slice(-maxHistory)) {
    const role: ChatMessage['role'] = item.role === 'assistant' ? 'assistant' : 'user';
    messages.push({ role, content: item.content });
  }

  messages.push({ role: 'user', content });

  const typingInterval = setInterval(() => {
    if (sourceMessage) {
      void sourceMessage.channel.sendTyping();
    }
  }, 9000);

  try {
    if (sourceMessage) {
      await sourceMessage.channel.sendTyping();
    }

    let responseMessage: Message | undefined;
    let accumulated = '';
    let lastUpdate = Date.now();

    const updateReply = async () => {
      if (interaction) {
        await interaction.editReply(accumulated || '...');
      } else if (responseMessage) {
        await responseMessage.edit(accumulated || '...');
      } else if (sourceMessage) {
        responseMessage = await sourceMessage.reply(accumulated || '...');
      }
    };

    const finalText = await streamChatCompletion({
      messages,
      model,
      onToken: async (token) => {
        accumulated += token;
        const now = Date.now();
        if (!responseMessage && !interaction && accumulated.length >= 3) {
          await updateReply();
        } else if (now - lastUpdate > 1000) {
          lastUpdate = now;
          await updateReply();
        }
      },
    });

    accumulated = finalText.trim();

    await updateReply();

    const finalContent = accumulated || 'I have nothing to add right now.';
    const chunks = splitMessage(finalContent);
    if (interaction) {
      const first = chunks.shift();
      if (first) {
        await interaction.editReply(first);
      }
      for (const chunk of chunks) {
        if (chunk.trim().length === 0) continue;
        await interaction.followUp(chunk);
      }
    } else if (responseMessage) {
      const first = chunks.shift();
      if (first) {
        await responseMessage.edit(first);
      }
      for (const chunk of chunks) {
        if (chunk.trim().length === 0) continue;
        await sourceMessage?.channel.send(chunk);
      }
    }

    if (finalContent.trim().length) {
      try {
        await memory.rememberMessage({
          channelId,
          content: finalContent.trim(),
          role: 'assistant',
          userId: client.user?.id ?? 'assistant',
        });
      } catch (error) {
        logger.error({ err: error }, 'Failed to store assistant response');
      }
    }

    recordRateUsage(userId, channelId);
  } catch (error) {
    logger.error({ err: error }, 'Failed to generate response');
    if (interaction) {
      await interaction.editReply('I ran into an error while thinking.');
    } else if (sourceMessage) {
      await sourceMessage.reply('I ran into an error while thinking.');
    }
  } finally {
    clearInterval(typingInterval);
  }
}

function getSystemPrompt(channelId: string, trendKeywords: string[], triggeredByTrend: boolean): string {
  const channelOverride = process.env[`SYSTEM_PROMPT_${channelId}`];
  const promptLines = [channelOverride ?? baseSystemPrompt];
  if (triggeredByTrend || Math.random() < trendManager.getTriggerChance()) {
    if (trendKeywords.length) {
      promptLines.push(
        `Trending topics right now: ${trendKeywords
          .map((keyword) => `"${keyword}"`)
          .join(', ')}`
      );
    }
  }
  promptLines.push('Use the provided memories to maintain continuity.');
  promptLines.push(brevityInstructions);
  return promptLines.join('\n');
}

function splitMessage(content: string, maxLength = 1900): string[] {
  if (content.length <= maxLength) return [content];
  const parts: string[] = [];
  let buffer = content;
  while (buffer.length > maxLength) {
    let sliceIndex = buffer.lastIndexOf('\n', maxLength);
    if (sliceIndex === -1) {
      sliceIndex = maxLength;
    }
    parts.push(buffer.slice(0, sliceIndex));
    buffer = buffer.slice(sliceIndex).trimStart();
  }
  if (buffer.length) {
    parts.push(buffer);
  }
  return parts;
}

function passesRateLimits(userId: string, channelId: string): boolean {
  const now = Date.now();
  const lastUserResponse = userCooldown.get(userId) ?? 0;
  if (now - lastUserResponse < 10_000) {
    return false;
  }

  const channelTimestamps = channelWindows.get(channelId) ?? [];
  const filtered = channelTimestamps.filter((timestamp) => now - timestamp < 30_000);
  if (filtered.length >= 5) {
    channelWindows.set(channelId, filtered);
    return false;
  }

  return true;
}

function recordRateUsage(userId: string, channelId: string) {
  const now = Date.now();
  userCooldown.set(userId, now);
  const channelTimestamps = channelWindows.get(channelId) ?? [];
  channelTimestamps.push(now);
  channelWindows.set(channelId, channelTimestamps);
}

async function handleShutdown() {
  logger.info('Shutting down gracefully...');
  trendManager.stop();
  await wait(500);
  client.destroy();
  process.exit(0);
}

process.on('SIGINT', handleShutdown);
process.on('SIGTERM', handleShutdown);

client.login(token).catch((error) => {
  logger.error({ err: error }, 'Failed to login to Discord');
  process.exit(1);
});
