"use client";

import Image from "next/image";
import { useState } from "react";

function HennaCornerMark() {
  return (
    <svg viewBox="0 0 28 28" className="w-5 h-5" fill="none">
      <circle cx="4.5" cy="4.5" r="2.8" stroke="currentColor" strokeWidth="0.7" opacity="0.4" />
      <circle cx="4.5" cy="4.5" r="1.1" fill="currentColor" opacity="0.55" />
      <line x1="8.2" y1="4.5" x2="15" y2="4.5" stroke="currentColor" strokeWidth="0.6" opacity="0.28" />
      <line x1="4.5" y1="8.2" x2="4.5" y2="15" stroke="currentColor" strokeWidth="0.6" opacity="0.28" />
      <circle cx="15" cy="4.5" r="0.9" fill="currentColor" opacity="0.32" />
      <circle cx="4.5" cy="15" r="0.9" fill="currentColor" opacity="0.32" />
    </svg>
  );
}

type Props = {
  name:     string;
  role:     string;
  image?:   string;
  rotation: string;
  size?:    "lg" | "sm";
};

export default function PolaroidCard({ name, role, image, rotation, size = "lg" }: Props) {
  const [imgError, setImgError] = useState(false);
  const showImage = image && !imgError;
  const firstName = name.split(" ")[0];

  const isLg     = size === "lg";
  const frame    = isLg ? "p-3 pb-14 shadow-[0_6px_28px_rgba(30,18,8,0.10)]" : "p-2.5 pb-11 shadow-[0_4px_20px_rgba(30,18,8,0.09)]";
  const aspect   = isLg ? "aspect-[3/4]" : "aspect-square";
  const nameSize = isLg ? "text-base"     : "text-sm";
  const nameGap  = isLg ? "mt-3"          : "mt-2";

  return (
    <div className={`group relative bg-white ${frame} ${rotation} hover:rotate-0 transition-all duration-500 ease-out cursor-default`}>

      {/* Henna corner marks on the white frame */}
      <div className="absolute top-2 left-2 text-henna/38 pointer-events-none">
        <HennaCornerMark />
      </div>
      <div className="absolute top-2 right-2 text-henna/38 rotate-90 pointer-events-none">
        <HennaCornerMark />
      </div>
      <div className="absolute bottom-2 right-2 text-henna/38 rotate-180 pointer-events-none">
        <HennaCornerMark />
      </div>
      <div className="absolute bottom-2 left-2 text-henna/38 -rotate-90 pointer-events-none">
        <HennaCornerMark />
      </div>

      {/* Photo area */}
      <div className={`relative overflow-hidden bg-gradient-to-br from-[#F7F2E8] to-[#EDE0CC] ${aspect}`}>
        {showImage ? (
          <Image
            src={image}
            alt={name}
            fill
            className="object-cover"
            onError={() => setImgError(true)}
          />
        ) : (
          /* Placeholder — shows ghost first name until photo is dropped in */
          <div className="absolute inset-0 flex flex-col justify-between p-3">
            <div className="flex justify-end">
              {/* subtle henna flourish inside photo corner */}
              <div className="text-henna/18 rotate-180">
                <HennaCornerMark />
              </div>
            </div>
            <span
              className="font-heading italic text-henna/18 leading-none select-none"
              style={{ fontSize: "clamp(2.2rem, 7vw, 3.5rem)" }}
            >
              {firstName}
            </span>
          </div>
        )}
      </div>

      {/* Name tag — polaroid handwritten label zone */}
      <div className={`text-center ${nameGap} px-2`}>
        <p className={`font-heading italic text-ink/75 leading-snug ${nameSize}`}>
          {name}
        </p>
        <p className="text-[9px] tracking-[0.22em] uppercase text-henna/50 mt-0.5">
          {role}
        </p>
      </div>
    </div>
  );
}
