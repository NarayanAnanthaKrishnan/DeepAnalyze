"use client";

import { createClient, type SupabaseClient } from "@supabase/supabase-js";

const SUPABASE_URL = process.env.NEXT_PUBLIC_SUPABASE_URL || "";
const SUPABASE_ANON_KEY = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY || "";

let client: SupabaseClient | null = null;

/** Returns the shared Supabase client, or null when auth isn't configured
 *  (e.g. local dev without env vars — gate is then bypassed). */
export function getSupabase(): SupabaseClient | null {
  if (!SUPABASE_URL || !SUPABASE_ANON_KEY) return null;
  if (!client) {
    client = createClient(SUPABASE_URL, SUPABASE_ANON_KEY, {
      auth: { persistSession: true, autoRefreshToken: true, detectSessionInUrl: true, flowType: "pkce" },
    });
  }
  return client;
}

export function isAuthConfigured(): boolean {
  return Boolean(SUPABASE_URL && SUPABASE_ANON_KEY);
}

/** Current verified email from a persisted Supabase session, if any. */
export async function getVerifiedEmail(): Promise<string | null> {
  const supabase = getSupabase();
  if (!supabase) return null;
  const { data } = await supabase.auth.getSession();
  const session = data.session;
  if (!session) return null;
  // Check expiry — supabase-js usually refreshes automatically on getSession,
  // but be defensive.
  if (session.expires_at && session.expires_at * 1000 < Date.now()) {
    const { data: refreshed } = await supabase.auth.refreshSession();
    if (!refreshed.session) return null;
  }
  return (session.user.email as string) || null;
}

export async function getAccessToken(): Promise<string | null> {
  const supabase = getSupabase();
  if (!supabase) return null;
  const { data } = await supabase.auth.getSession();
  return data.session?.access_token ?? null;
}

export function signOutLocally(): void {
  getSupabase()?.auth.signOut();
}
