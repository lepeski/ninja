import fetch from 'node-fetch';
import logger from './logger.js';

export interface TrendOptions {
  woeid: number;
  intervalMs: number;
  bearerToken?: string;
  triggerChance: number;
  keywordChance: number;
}

export class TrendManager {
  private keywords: string[] = [];
  private timer?: NodeJS.Timeout;

  constructor(private readonly options: TrendOptions) {}

  start(): void {
    this.stop();
    void this.updateTrends();
    this.timer = setInterval(() => void this.updateTrends(), this.options.intervalMs);
  }

  stop(): void {
    if (this.timer) {
      clearInterval(this.timer);
      this.timer = undefined;
    }
  }

  getKeywords(): string[] {
    return this.keywords;
  }

  getTriggerChance(): number {
    return this.options.triggerChance;
  }

  getKeywordChance(): number {
    return this.options.keywordChance;
  }

  private async updateTrends(): Promise<void> {
    if (!this.options.bearerToken) {
      logger.warn('X_BEARER_TOKEN is not configured; trending keywords disabled.');
      this.keywords = [];
      return;
    }

    try {
      const response = await fetch(
        `https://api.x.com/1.1/trends/place.json?id=${this.options.woeid}`,
        {
          headers: {
            Authorization: `Bearer ${this.options.bearerToken}`,
          },
        }
      );

      if (!response.ok) {
        logger.warn({ status: response.status, statusText: response.statusText }, 'Failed to fetch trends');
        return;
      }

      const data = (await response.json()) as Array<{
        trends: Array<{ name: string }>;
      }>;

      const trends = data?.[0]?.trends ?? [];
      this.keywords = trends.slice(0, 10).map((trend) => trend.name.toLowerCase());
      logger.info({ keywords: this.keywords }, 'Updated trending keywords');
    } catch (error) {
      logger.error({ err: error }, 'Error updating trends');
    }
  }
}

export function parseDuration(input: string | undefined, fallbackMs: number): number {
  if (!input) return fallbackMs;
  const match = input.trim().match(/^(\d+)(ms|s|m|h|d)?$/i);
  if (!match) return fallbackMs;
  const value = Number(match[1]);
  const unit = match[2]?.toLowerCase() ?? 'ms';
  const unitMap: Record<string, number> = {
    ms: 1,
    s: 1000,
    m: 60 * 1000,
    h: 60 * 60 * 1000,
    d: 24 * 60 * 60 * 1000,
  };
  return value * (unitMap[unit] ?? 1);
}
