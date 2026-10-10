// Keep the development proxy as the default, while allowing the static Vercel
// build to call a separately hosted API. Normalize a trailing slash so paths
// are not accidentally joined with `//api/...`.
export const API = (import.meta.env.VITE_API_URL || '').trim().replace(/\/+$/, '');
export async function get(path: string) {
  const r = await fetch(API + path);
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}
export async function upload(
  path: string,
  file: File,
  fields: Record<string, string>,
  attachments: Record<string, File | null> = {},
) {
  const f = new FormData();
  f.append('file', file);
  Object.entries(fields).forEach(([k, v]) => f.append(k, v));
  Object.entries(attachments).forEach(([k, v]) => v && f.append(k, v));
  const r = await fetch(API + path, { method: 'POST', body: f });
  const j = await r.json();
  if (!r.ok) throw new Error(j.detail || 'Prediction failed');
  return j;
}
