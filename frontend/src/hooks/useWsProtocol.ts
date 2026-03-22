import { useMemo } from 'react';

import type { BackendState } from '../types/backendState';
import type { BackendSocketMessage, WsDeltaMessage, WsEventMessage, WsSnapshotMessage, WsV2Envelope } from '../types/wsProtocol';

function isObject(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function deepMerge<T>(base: T, patch: Partial<T>): T {
  if (!isObject(base) || !isObject(patch)) {
    return patch as T;
  }

  const merged: Record<string, unknown> = { ...base };
  for (const [key, value] of Object.entries(patch)) {
    if (value === null) {
      merged[key] = null;
      continue;
    }
    const current = merged[key];
    merged[key] = isObject(current) && isObject(value)
      ? deepMerge(current, value)
      : value;
  }
  return merged as T;
}

export function isWsV2Envelope(message: unknown): message is WsV2Envelope {
  return isObject(message) && message.v === 2 && typeof message.t === 'string' && isObject(message.payload);
}

export function useWsProtocol(rawMessage: BackendSocketMessage | null, previousState: BackendState | null) {
  return useMemo(() => {
    if (!rawMessage) {
      return {
        state: previousState,
        lastEvent: null as WsEventMessage | null,
        protocolVersion: 1,
      };
    }

    if (!isWsV2Envelope(rawMessage)) {
      return {
        state: rawMessage as BackendState,
        lastEvent: null as WsEventMessage | null,
        protocolVersion: 1,
      };
    }

    const v2Message = rawMessage as WsSnapshotMessage | WsDeltaMessage | WsEventMessage;
    switch (v2Message.t) {
      case 'snapshot':
        return {
          state: v2Message.payload,
          lastEvent: null as WsEventMessage | null,
          protocolVersion: 2,
        };
      case 'delta': {
        const mergedState = previousState ? deepMerge(previousState, v2Message.payload) : previousState;
        return {
          state: mergedState,
          lastEvent: null as WsEventMessage | null,
          protocolVersion: 2,
        };
      }
      case 'event':
      default:
        return {
          state: previousState,
          lastEvent: v2Message,
          protocolVersion: 2,
        };
    }
  }, [previousState, rawMessage]);
}
