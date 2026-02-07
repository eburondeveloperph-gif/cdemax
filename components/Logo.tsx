
/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
*/
import React from 'react';

export const Logo: React.FC<{ className?: string }> = ({ className = "w-8 h-8" }) => {
  return (
    <div className={`${className} relative flex items-center justify-center`}>
      <img 
        src="https://eburon.ai/assets/icon-eburon.png" 
        alt="Eburon Logo" 
        className="w-full h-full object-contain"
      />
    </div>
  );
};
