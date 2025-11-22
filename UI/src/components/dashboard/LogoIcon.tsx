import React from "react";
export function LogoIcon({ className = "w-10 h-10" }: { className?: string }) {
  return (
    <svg
      viewBox="0 0 100 100"
      className={className}
      xmlns="http://www.w3.org/2000/svg"
    >
      <defs>
        <linearGradient id="gearGradient" x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%" style={{ stopColor: "#60a5fa", stopOpacity: 1 }} />
          <stop
            offset="100%"
            style={{ stopColor: "#3b82f6", stopOpacity: 1 }}
          />
        </linearGradient>
      </defs>

      {/* Main gear body */}
      <path
        d="M50 15 L55 15 L57 10 L60 10 L62 15 L67 15 L70 20 L75 20 L77 23 L80 25 L80 30 L85 32 L85 37 L90 40 L90 43 L87 45 L90 47 L90 50 L87 52 L90 55 L90 60 L85 62 L85 67 L80 70 L80 75 L77 77 L75 80 L70 80 L67 85 L62 85 L60 90 L57 90 L55 85 L50 85 L45 85 L43 90 L40 90 L38 85 L33 85 L30 80 L25 80 L23 77 L20 75 L20 70 L15 67 L15 62 L10 60 L10 55 L13 52 L10 50 L10 47 L13 45 L10 43 L10 40 L15 37 L15 32 L20 30 L20 25 L23 23 L25 20 L30 20 L33 15 L38 15 L40 10 L43 10 L45 15 Z"
        fill="url(#gearGradient)"
        stroke="#1e40af"
        strokeWidth="1"
      />

      {/* Inner circle */}
      <circle
        cx="50"
        cy="50"
        r="20"
        fill="#0f172a"
        stroke="#3b82f6"
        strokeWidth="2"
      />

      {/* Circuit pattern in center */}
      <circle cx="50" cy="50" r="4" fill="#60a5fa" />
      <line
        x1="50"
        y1="46"
        x2="50"
        y2="38"
        stroke="#60a5fa"
        strokeWidth="1.5"
      />
      <line
        x1="50"
        y1="54"
        x2="50"
        y2="62"
        stroke="#60a5fa"
        strokeWidth="1.5"
      />
      <line
        x1="46"
        y1="50"
        x2="38"
        y2="50"
        stroke="#60a5fa"
        strokeWidth="1.5"
      />
      <line
        x1="54"
        y1="50"
        x2="62"
        y2="50"
        stroke="#60a5fa"
        strokeWidth="1.5"
      />

      {/* Circuit nodes */}
      <circle cx="50" cy="38" r="2" fill="#60a5fa" />
      <circle cx="50" cy="62" r="2" fill="#60a5fa" />
      <circle cx="38" cy="50" r="2" fill="#60a5fa" />
      <circle cx="62" cy="50" r="2" fill="#60a5fa" />

      {/* Diagonal circuit lines */}
      <line
        x1="44"
        y1="44"
        x2="38"
        y2="38"
        stroke="#3b82f6"
        strokeWidth="1"
        opacity="0.6"
      />
      <line
        x1="56"
        y1="44"
        x2="62"
        y2="38"
        stroke="#3b82f6"
        strokeWidth="1"
        opacity="0.6"
      />
      <line
        x1="44"
        y1="56"
        x2="38"
        y2="62"
        stroke="#3b82f6"
        strokeWidth="1"
        opacity="0.6"
      />
      <line
        x1="56"
        y1="56"
        x2="62"
        y2="62"
        stroke="#3b82f6"
        strokeWidth="1"
        opacity="0.6"
      />

      {/* Corner circuit nodes */}
      <circle cx="38" cy="38" r="1.5" fill="#3b82f6" />
      <circle cx="62" cy="38" r="1.5" fill="#3b82f6" />
      <circle cx="38" cy="62" r="1.5" fill="#3b82f6" />
      <circle cx="62" cy="62" r="1.5" fill="#3b82f6" />
    </svg>
  );
}
