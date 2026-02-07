
/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
*/
import React, { useEffect, useState } from 'react';
import { 
  ArrowDownTrayIcon, 
  PlusIcon, 
  CodeBracketIcon, 
  XMarkIcon, 
  ComputerDesktopIcon, 
  DevicePhoneMobileIcon, 
  DeviceTabletIcon, 
  ShieldCheckIcon, 
  CheckBadgeIcon 
} from '@heroicons/react/24/outline';
import { Creation } from './CreationHistory';

interface LivePreviewProps {
  creation: Creation | null;
  isLoading: boolean;
  isFocused: boolean;
  onReset: () => void;
  onVerify?: () => void;
}

type ViewportMode = 'desktop' | 'tablet' | 'phone';

export const LivePreview: React.FC<LivePreviewProps> = ({ creation, isLoading, isFocused, onReset, onVerify }) => {
    const [status, setStatus] = useState("Initializing...");
    const [viewMode, setViewMode] = useState<ViewportMode>('desktop');

    useEffect(() => {
        if (isLoading) {
            const statuses = ["Injecting...", "Compiling...", "Logic Check...", "Verifying..."];
            let i = 0;
            const interval = setInterval(() => {
                setStatus(statuses[i % statuses.length]);
                i++;
            }, 1000);
            return () => clearInterval(interval);
        }
    }, [isLoading]);

    const handleExport = () => {
        if (!creation) return;
        const blob = new Blob([creation.html], { type: "text/html" });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `verified_build_${Date.now()}.html`;
        a.click();
    };

    const getViewportWidth = () => {
        switch(viewMode) {
            case 'phone': return 'max-w-[390px]';
            case 'tablet': return 'max-w-[768px]';
            default: return 'max-w-full';
        }
    };

  return (
    <div
      className={`
        fixed z-50 flex flex-col
        rounded-t-[6px] md:rounded-[6px] overflow-hidden border border-zinc-200 dark:border-white/10 bg-white dark:bg-[#080808] shadow-[0_0_80px_rgba(0,0,0,0.3)] dark:shadow-[0_0_80px_rgba(0,0,0,1)]
        transition-all duration-500 ease-in-out
        ${isFocused
          ? 'inset-0 md:inset-6 opacity-100'
          : 'top-[110%] left-0 w-full h-full opacity-0 pointer-events-none'
        }
      `}
    >
      {/* Header */}
      <div className="bg-zinc-50 dark:bg-[#0f0f0f] px-6 py-3 flex items-center justify-between border-b border-zinc-200 dark:border-white/5 shrink-0">
        <div className="flex items-center space-x-6">
           <div className="flex space-x-1.5">
                <button onClick={onReset} className="w-3 h-3 rounded-full bg-red-500/50 hover:bg-red-500 transition-all flex items-center justify-center group shadow-sm">
                    <XMarkIcon className="w-1.5 h-1.5 text-white opacity-0 group-hover:opacity-100" />
                </button>
                <div className="w-3 h-3 rounded-full bg-zinc-200 dark:bg-zinc-800 shadow-sm"></div>
                <div className="w-3 h-3 rounded-full bg-zinc-200 dark:bg-zinc-800 shadow-sm"></div>
           </div>
           
           <div className="h-4 w-px bg-zinc-200 dark:bg-white/10 mx-2"></div>
           
           <div className="flex bg-zinc-200/50 dark:bg-white/5 rounded-[6px] p-0.5 border border-zinc-200 dark:border-white/10">
                <button onClick={() => setViewMode('desktop')} className={`p-1.5 rounded-[4px] transition-all ${viewMode === 'desktop' ? 'bg-white text-black shadow-sm' : 'text-zinc-400 hover:text-zinc-900 dark:hover:text-zinc-300'}`}><ComputerDesktopIcon className="w-3.5 h-3.5" /></button>
                <button onClick={() => setViewMode('tablet')} className={`p-1.5 rounded-[4px] transition-all ${viewMode === 'tablet' ? 'bg-white text-black shadow-sm' : 'text-zinc-400 hover:text-zinc-900 dark:hover:text-zinc-300'}`}><DeviceTabletIcon className="w-3.5 h-3.5" /></button>
                <button onClick={() => setViewMode('phone')} className={`p-1.5 rounded-[4px] transition-all ${viewMode === 'phone' ? 'bg-white text-black shadow-sm' : 'text-zinc-400 hover:text-zinc-900 dark:hover:text-zinc-300'}`}><DevicePhoneMobileIcon className="w-3.5 h-3.5" /></button>
           </div>
        </div>

        <div className="flex items-center space-x-2">
            {!isLoading && creation && (
                <>
                    <button onClick={onVerify} className="flex items-center space-x-1.5 text-[10px] font-bold text-emerald-600 dark:text-emerald-400 border border-emerald-200 dark:border-emerald-500/20 bg-emerald-50 dark:bg-emerald-500/5 hover:bg-emerald-100 dark:hover:bg-emerald-500/10 px-3 py-1.5 rounded-[6px] transition-all uppercase tracking-tighter shadow-sm"><ShieldCheckIcon className="w-3.5 h-3.5" /><span>Verify</span></button>
                    <button onClick={handleExport} className="p-2 text-zinc-500 hover:text-zinc-900 dark:hover:text-white rounded-[6px] transition-all"><ArrowDownTrayIcon className="w-4 h-4" /></button>
                    <button onClick={onReset} className="flex items-center space-x-1.5 text-[10px] font-bold bg-zinc-900 dark:bg-white text-white dark:text-black px-4 py-1.5 rounded-[6px] transition-all hover:opacity-90 uppercase tracking-tighter shadow-sm"><span>Reset</span></button>
                </>
            )}
        </div>
      </div>

      {/* Sandbox */}
      <div className="relative flex-1 bg-zinc-100 dark:bg-[#121212] flex flex-col items-center overflow-hidden p-4 md:p-6">
        {isLoading ? (
          <div className="absolute inset-0 flex flex-col items-center justify-center bg-white dark:bg-[#080808]">
             <CodeBracketIcon className="w-12 h-12 text-zinc-300 dark:text-zinc-800 animate-pulse mb-4" />
             <p className="text-[10px] font-bold text-zinc-400 dark:text-zinc-500 uppercase tracking-[0.3em]">{status}</p>
          </div>
        ) : creation?.html ? (
            <div className={`h-full w-full ${getViewportWidth()} bg-white shadow-2xl rounded-[6px] overflow-hidden transition-all duration-500 border border-zinc-200 dark:border-white/5`}>
                <iframe title="Sandbox" srcDoc={creation.html} className="w-full h-full border-none" sandbox="allow-scripts allow-forms allow-popups allow-modals allow-same-origin" />
            </div>
        ) : (
            <div className="h-full flex flex-col items-center justify-center text-zinc-400 dark:text-zinc-700 text-[10px] uppercase tracking-[0.3em] italic">Awaiting Build</div>
        )}
      </div>

      {/* Status Bar */}
      {creation && !isLoading && (
          <div className="bg-zinc-50 dark:bg-[#0f0f0f] border-t border-zinc-200 dark:border-white/5 px-6 py-2 flex items-center justify-between shrink-0">
             <div className="flex items-center space-x-2">
                <CheckBadgeIcon className="w-3 h-3 text-emerald-500" />
                <span className="text-[9px] font-bold text-zinc-400 dark:text-zinc-600 uppercase tracking-widest">Verified Integrity</span>
             </div>
             <div className="text-[9px] font-mono text-zinc-400 dark:text-zinc-700 uppercase">{creation.id.split('-')[0]}</div>
          </div>
      )}
    </div>
  );
};
