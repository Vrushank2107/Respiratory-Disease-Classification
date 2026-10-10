import { useEffect, useState } from 'react';

/** Create a browser object URL and revoke it when its source changes or unmounts. */
export function useObjectUrl(blob: Blob | null | undefined) {
  const [url, setUrl] = useState('');
  useEffect(() => {
    if (!blob) { setUrl(''); return; }
    const nextUrl = URL.createObjectURL(blob);
    setUrl(nextUrl);
    return () => URL.revokeObjectURL(nextUrl);
  }, [blob]);
  return url;
}
