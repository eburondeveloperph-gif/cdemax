
/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
*/
import React from 'react';
import { ClockIcon, ArrowRightIcon, DocumentIcon, PhotoIcon } from '@heroicons/react/24/outline';

export interface Creation {
  id: string;
  name: string;
  html: string;
  originalImage?: string; // Base64 data URL
  timestamp: Date;
}

interface CreationHistoryProps {
  history: Creation[];
  onSelect: (creation: Creation) => void;
}

export const CreationHistory: React.FC<CreationHistoryProps> = ({ history, onSelect }) => {
  if (history.length === 0) return (
    <div className="p-4 text-center text-zinc-400 dark:text-zinc-600 text-[11px] uppercase tracking-widest italic">No deployments found.</div>
  );

  return (
    <div className="w-full space-y-3">
      {history.map((item) => {
        const isPdf = item.originalImage?.startsWith('data:application/pdf');
        return (
          <button
            key={item.id}
            onClick={() => onSelect(item)}
            className="group w-full relative flex flex-col text-left p-4 bg-zinc-50 dark:bg-white/5 border border-zinc-200 dark:border-white/5 rounded-[6px] hover:border-zinc-300 dark:hover:border-white/20 hover:bg-zinc-100 dark:hover:bg-white/[0.08] transition-all duration-200 overflow-hidden shadow-sm"
          >
            <div className="flex items-start justify-between w-full mb-3">
              <div className="p-2 bg-white dark:bg-black/40 rounded-[4px] border border-zinc-200 dark:border-white/5 shadow-sm">
                  {isPdf ? (
                      <DocumentIcon className="w-3.5 h-3.5 text-zinc-400 dark:text-zinc-500" />
                  ) : item.originalImage ? (
                      <PhotoIcon className="w-3.5 h-3.5 text-zinc-400 dark:text-zinc-500" />
                  ) : (
                      <DocumentIcon className="w-3.5 h-3.5 text-zinc-400 dark:text-zinc-500" />
                  )}
              </div>
              <span className="text-[9px] font-mono text-zinc-400 dark:text-zinc-600">
                {item.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
              </span>
            </div>
            
            <h3 className="text-xs font-semibold text-zinc-600 dark:text-zinc-400 group-hover:text-black dark:group-hover:text-white truncate">
              {item.name}
            </h3>
            <div className="flex items-center space-x-1 mt-1.5 opacity-0 group-hover:opacity-100 transition-opacity">
              <span className="text-[9px] font-bold text-blue-600 dark:text-blue-400 uppercase tracking-tighter">Load Architecture</span>
              <ArrowRightIcon className="w-2.5 h-2.5 text-blue-600 dark:text-blue-400" />
            </div>
          </button>
        );
      })}
    </div>
  );
};
