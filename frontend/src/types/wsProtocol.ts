import type { BackendState } from './backendState';

export interface WsV2Envelope<TPayload = Record<string, unknown>> {
  v: 2;
  t: 'snapshot' | 'delta' | 'event';
  seq: number;
  ts_ms: number;
  payload: TPayload;
  base_seq?: number;
}

export interface WsSnapshotMessage extends WsV2Envelope<BackendState> {
  t: 'snapshot';
}

export interface WsDeltaMessage extends WsV2Envelope<Partial<BackendState>> {
  t: 'delta';
  base_seq?: number;
}

export interface WsEventPayload {
  event: string;
  [key: string]: unknown;
}

export interface WsEventMessage extends WsV2Envelope<WsEventPayload> {
  t: 'event';
}

export type BackendSocketMessage = BackendState | WsSnapshotMessage | WsDeltaMessage | WsEventMessage;
