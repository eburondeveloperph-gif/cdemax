
/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
*/
import { GoogleGenAI, GenerateContentResponse, Chat } from "@google/genai";

export const MODELS = {
  CODEMAX_13: 'gemini-3-flash-preview',
  CODEMAX_PRO: 'gemini-3-pro-preview',
  CODEMAX_BETA: 'gemini-3-pro-preview',
  GEMMA_3: 'gemini-3-flash-preview' // Using Flash-3 as the proxy for high-speed Gemma 3 tasks
};

export interface Message {
  role: 'user' | 'model';
  parts: { text?: string; inlineData?: { data: string; mimeType: string } }[];
  modelName?: string;
}

const SYSTEM_INSTRUCTION = `You are the Elite CodeMax Software Architect. 
Your output must consist ONLY of the requested source code. 
DO NOT provide any reasoning, explanations, conversational filler, or introductions. 
Return complete, production-ready, standalone HTML files including all necessary CSS and JavaScript. 
Never truncate code. Never hesitate. Never ask follow-up questions. 
If the user provides a prompt, translate it directly into the most efficient and visually stunning code possible. 
YOUR OUTPUT IS THE RAW SOURCE CODE ONLY.`;

export async function chatStream(
  modelName: string,
  history: Message[],
  onChunk: (text: string) => void
) {
  const aiClient = new GoogleGenAI({ apiKey: process.env.API_KEY });
  
  const chat = aiClient.chats.create({
    model: modelName,
    config: {
      systemInstruction: SYSTEM_INSTRUCTION,
      thinkingConfig: { thinkingBudget: 0 } // Disable thinking/reasoning for direct code output
    }
  });

  const lastMessage = history[history.length - 1];
  
  const response = await chat.sendMessageStream({
    message: lastMessage.parts
  });

  let fullText = "";
  for await (const chunk of response) {
    const chunkText = chunk.text || "";
    fullText += chunkText;
    onChunk(fullText);
  }
  return fullText;
}

export async function chatOllamaStream(
  url: string,
  modelName: string,
  history: Message[],
  onChunk: (text: string) => void
) {
  const messages = history.map(msg => ({
    role: msg.role === 'model' ? 'assistant' : 'user',
    content: msg.parts.map(p => p.text).join('\n')
  }));

  const response = await fetch(`${url}/api/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      model: modelName,
      messages: [
        { role: 'system', content: SYSTEM_INSTRUCTION },
        ...messages
      ],
      stream: true
    })
  });

  if (!response.body) throw new Error("Ollama stream failed");

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let fullText = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    
    const chunk = decoder.decode(value, { stream: true });
    const lines = chunk.split('\n');
    
    for (const line of lines) {
      if (!line) continue;
      try {
        const json = JSON.parse(line);
        if (json.message?.content) {
          fullText += json.message.content;
          onChunk(fullText);
        }
      } catch (e) {
        // Handle partial JSON
      }
    }
  }
  return fullText;
}
