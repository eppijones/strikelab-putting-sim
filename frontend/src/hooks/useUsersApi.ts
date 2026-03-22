import { useCallback, useEffect, useRef, useState } from 'react';

import type { User } from '../types/backendState';

export function useUsersApi(httpBaseUrl: string, backendReady: boolean, isConnected: boolean) {
  const [users, setUsers] = useState<User[]>([]);
  const hasLoggedUserFetchOfflineRef = useRef(false);

  const refreshUsers = useCallback(async () => {
    try {
      const response = await fetch(`${httpBaseUrl}/api/users`);
      if (response.ok) {
        const data = await response.json();
        if (data.success) {
          setUsers(data.users);
          hasLoggedUserFetchOfflineRef.current = false;
        }
      }
    } catch {
      if (!hasLoggedUserFetchOfflineRef.current) {
        console.warn('Backend not reachable yet; user list will load automatically when connected.');
        hasLoggedUserFetchOfflineRef.current = true;
      }
    }
  }, [httpBaseUrl]);

  const createUser = useCallback(async (name: string, handicap: number) => {
    try {
      const response = await fetch(`${httpBaseUrl}/api/users`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name, handicap }),
      });
      if (response.ok) {
        await refreshUsers();
      }
    } catch (error) {
      console.error('Error creating user:', error);
    }
  }, [httpBaseUrl, refreshUsers]);

  const deleteUser = useCallback(async (userId: number) => {
    try {
      const response = await fetch(`${httpBaseUrl}/api/users/${userId}`, {
        method: 'DELETE',
      });
      if (response.ok) {
        await refreshUsers();
      }
    } catch (error) {
      console.error('Error deleting user:', error);
    }
  }, [httpBaseUrl, refreshUsers]);

  const resetUserData = useCallback(async (userId: number): Promise<{ success: boolean; shots_deleted?: number; sessions_deleted?: number }> => {
    try {
      const response = await fetch(`${httpBaseUrl}/api/users/${userId}/reset`, {
        method: 'POST',
      });
      if (response.ok) {
        const data = await response.json();
        return {
          success: true,
          shots_deleted: data.shots_deleted,
          sessions_deleted: data.sessions_deleted,
        };
      }
      return { success: false };
    } catch (error) {
      console.error('Error resetting user data:', error);
      return { success: false };
    }
  }, [httpBaseUrl]);

  const selectUser = useCallback(async (userId: number | null) => {
    try {
      const response = await fetch(`${httpBaseUrl}/api/session/user`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ user_id: userId }),
      });
      if (!response.ok) {
        throw new Error('Failed to select user');
      }
    } catch (error) {
      console.error('Error selecting user:', error);
    }
  }, [httpBaseUrl]);

  useEffect(() => {
    if (backendReady) {
      refreshUsers();
    }
  }, [backendReady, refreshUsers]);

  useEffect(() => {
    if (isConnected) {
      refreshUsers();
    }
  }, [isConnected, refreshUsers]);

  return {
    users,
    refreshUsers,
    createUser,
    deleteUser,
    resetUserData,
    selectUser,
  };
}
