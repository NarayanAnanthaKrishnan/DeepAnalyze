"use client";

import {
  PromptInput,
  PromptInputAction,
  PromptInputActions,
  PromptInputTextarea,
} from "@/components/ui/prompt-input";
import { Button } from "@/components/ui/button";
import { ThemeSelector } from "@/components/theme-selector";
import {
  ArrowUp,
  Paperclip,
  Mic,
  Square,
  X,
  Zap,
} from "lucide-react";
import { useRef } from "react";

interface PromptInputEnhancedProps {
  input: string;
  onInputChange: (value: string) => void;
  files: File[];
  onFilesChange: (files: File[]) => void;
  reportTheme: string;
  onReportThemeChange: (theme: string) => void;
  isLoading: boolean;
  onSubmit: () => void;
}

export function PromptInputEnhanced({
  input,
  onInputChange,
  files,
  onFilesChange,
  reportTheme,
  onReportThemeChange,
  isLoading,
  onSubmit,
}: PromptInputEnhancedProps) {
  const uploadInputRef = useRef<HTMLInputElement>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      const newFiles = Array.from(e.target.files);
      onFilesChange([...files, ...newFiles]);
    }
  };

  const handleRemoveFile = (index: number) => {
    onFilesChange(files.filter((_, i) => i !== index));
    if (uploadInputRef?.current) {
      uploadInputRef.current.value = "";
    }
  };

  return (
    <PromptInput
      value={input}
      onValueChange={onInputChange}
      isLoading={isLoading}
      onSubmit={onSubmit}
      className="w-full flex flex-col gap-1 !rounded-none !border-0 !bg-transparent !p-0 !shadow-none ring-0"
    >
      {files.length > 0 && (
        <div className="flex flex-wrap gap-1.5 pb-0">
          {files.map((file, index) => (
            <div
              key={index}
              className="flex items-center gap-2 px-2 py-1 bg-primary/5 border border-primary/20 text-primary font-mono text-[8px] uppercase tracking-widest backdrop-blur-md"
              onClick={(e) => e.stopPropagation()}
            >
              <Paperclip className="size-2.5" />
              <span className="max-w-[140px] truncate">{file.name}</span>
              <button
                onClick={() => handleRemoveFile(index)}
                className="hover:text-destructive transition-colors ml-1"
              >
                <X className="size-2.5" />
              </button>
            </div>
          ))}
        </div>
      )}

      <div className="relative group w-full">
        <div className="absolute left-0 top-[14px] sm:top-[20px] md:top-[24px] w-1 h-5 sm:h-7 md:h-8 bg-primary/40 hidden sm:block opacity-50 group-focus-within:opacity-100 transition-opacity rounded-full" />
        <PromptInputTextarea
          placeholder="what shall we discover today?"
          className="dark:bg-transparent !text-xl sm:!text-2xl md:!text-3xl !font-display !font-medium !tracking-tight placeholder:text-muted-foreground/30 !min-h-[50px] sm:!min-h-[70px] !pl-0 sm:!pl-6 !pr-0 !pt-[12px] sm:!pt-[16px] md:!pt-[20px] !pb-4 focus-visible:!ring-0 transition-all focus:placeholder:opacity-0 caret-primary w-full"
        />
      </div>

      <PromptInputActions className="flex flex-wrap items-center justify-between border-t border-border/40 pt-3 mt-1 gap-y-3">
        <div className="flex items-center flex-wrap gap-0.5 flex-shrink-0">
          <PromptInputAction tooltip="Attach Dataset [CSV, PDF, etc]">
            <label
              htmlFor="file-upload"
              className="group flex size-8 sm:size-8 cursor-pointer items-center justify-center border border-border/20 bg-secondary/30 hover:bg-secondary/60 transition-all rounded-none flex-shrink-0"
            >
              <input
                ref={uploadInputRef}
                type="file"
                multiple
                onChange={handleFileChange}
                className="hidden"
                id="file-upload"
              />
              <Paperclip className="text-primary size-3" />
            </label>
          </PromptInputAction>

          <div className="w-px h-6 bg-border/40 mx-px sm:mx-0.5" />

          <ThemeSelector
            value={reportTheme}
            onValueChange={onReportThemeChange}
          />

          <div className="w-px h-6 bg-border/40 mx-px sm:mx-0.5" />

          <div className="flex items-center gap-1.5 h-8 px-2.5 border border-blue-500/30 bg-blue-500/10 text-blue-600 dark:text-blue-400">
            <Zap className="size-3 text-blue-500 shrink-0" />
            <span className="font-mono text-[8px] sm:text-[9px] uppercase tracking-[0.1em] font-bold">Gemini 3 Flash</span>
          </div>
        </div>

        <div className="flex items-center gap-1.5 sm:gap-2 flex-shrink-0 ml-auto sm:ml-0">
          <PromptInputAction tooltip="Voice Command [Inactive]">
            <button
              className="size-8 sm:size-8 flex items-center justify-center text-muted-foreground hover:text-foreground hover:bg-secondary/50 transition-colors border border-transparent hover:border-border/20 rounded-none flex-shrink-0"
              onClick={(e) => e.stopPropagation()}
            >
              <Mic className="size-3" />
            </button>
          </PromptInputAction>

          <PromptInputAction
            tooltip={isLoading ? "Halt Synthesis" : "Execute Analysis"}
          >
            <Button
              variant="default"
              size="sm"
              className="h-8 sm:h-8 rounded-none px-3 sm:px-4 font-mono text-[8px] sm:text-[9px] uppercase tracking-[0.2em] bg-foreground text-background hover:bg-primary hover:text-primary-foreground transition-all shadow-lg shadow-primary/10 whitespace-nowrap"
              onClick={onSubmit}
            >
              {isLoading ? (
                <div className="flex items-center gap-1.5">
                  <Square className="size-3 fill-current" />
                  <span>Halt</span>
                </div>
              ) : (
                <div className="flex items-center gap-1.5">
                  <span className="mt-0.5">Execute</span>
                  <ArrowUp className="size-3" />
                </div>
              )}
            </Button>
          </PromptInputAction>
        </div>
      </PromptInputActions>
    </PromptInput>
  );
}
