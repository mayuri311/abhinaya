export default function HennaDivider({ className = "" }: { className?: string }) {
  return (
    <svg
      viewBox="0 0 200 12"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      className={className}
      aria-hidden="true"
    >
      <line x1="0" y1="6" x2="82" y2="6" stroke="currentColor" strokeWidth="0.5" />
      <circle cx="90" cy="6" r="2" fill="currentColor" />
      <path d="M100 2L104 6L100 10L96 6Z" fill="currentColor" />
      <circle cx="110" cy="6" r="2" fill="currentColor" />
      <line x1="118" y1="6" x2="200" y2="6" stroke="currentColor" strokeWidth="0.5" />
    </svg>
  );
}
