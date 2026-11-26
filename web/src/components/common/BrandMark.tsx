export function BrandMark({ size = 28 }: { size?: number }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 64 64"
      fill="none"
      aria-hidden
    >
      <rect width="64" height="64" rx="16" fill="currentColor" />
      <path
        d="M18 40V24h8.5c5.2 0 8.5 2.8 8.5 7.2 0 4.5-3.4 7.3-8.6 7.3H24v1.5h10.8V44H18zm6-9.2h2.3c2.4 0 3.9-1.2 3.9-3.2S28.7 24.5 26.3 24.5H24V30.8zM38 44l7.2-20h6.1L58.5 44h-6.2l-1.2-3.7H45.4L44.2 44H38zm8.4-8.4h5.3l-2.6-8.1-2.7 8.1z"
        fill="var(--bg)"
      />
    </svg>
  );
}
