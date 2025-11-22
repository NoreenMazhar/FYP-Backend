"use client";

import * as React from "react";
import { Dialog as HeadlessDialog, DialogPanel, DialogBackdrop, DialogTitle, DialogDescription } from "@headlessui/react";
import { XIcon } from "lucide-react";

import { cn } from "./utils";

interface DialogContextValue {
  onOpenChange?: (open: boolean) => void;
}

const DialogContext = React.createContext<DialogContextValue>({});

interface DialogProps {
  open?: boolean;
  onOpenChange?: (open: boolean) => void;
  children: React.ReactNode;
  preventAutoClose?: boolean;
}

function Dialog({ open, onOpenChange, children, preventAutoClose, ...props }: DialogProps) {
  return (
    <DialogContext.Provider value={{ onOpenChange }}>
      <HeadlessDialog
        open={open ?? false}
        onClose={() => {
          // Only close if not prevented
          if (!preventAutoClose) {
            onOpenChange?.(false);
          }
        }}
        autoFocus={false}
        {...props}
      >
        {children}
      </HeadlessDialog>
    </DialogContext.Provider>
  );
}

interface DialogTriggerProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  asChild?: boolean;
}

const DialogTrigger = React.forwardRef<HTMLButtonElement, DialogTriggerProps>(
  ({ asChild, children, onClick, ...props }, ref) => {
    const { onOpenChange } = React.useContext(DialogContext);

    const handleClick = (e: React.MouseEvent<HTMLButtonElement>) => {
      onOpenChange?.(true);
      onClick?.(e);
    };

    if (asChild && React.isValidElement(children)) {
      return React.cloneElement(children, {
        ...props,
        ref,
        onClick: (e: React.MouseEvent) => {
          handleClick(e as any);
          children.props.onClick?.(e);
        },
      } as any);
    }
    return (
      <button ref={ref} onClick={handleClick} {...props}>
        {children}
      </button>
    );
  }
);

DialogTrigger.displayName = "DialogTrigger";

function DialogPortal({ children }: { children: React.ReactNode }) {
  return <>{children}</>;
}

function DialogClose({ children, onClick, ...props }: React.ButtonHTMLAttributes<HTMLButtonElement>) {
  const { onOpenChange } = React.useContext(DialogContext);

  const handleClick = (e: React.MouseEvent<HTMLButtonElement>) => {
    onOpenChange?.(false);
    onClick?.(e);
  };

  return (
    <button type="button" onClick={handleClick} {...props}>
      {children}
    </button>
  );
}

const DialogOverlay = React.forwardRef<
  HTMLDivElement,
  React.HTMLAttributes<HTMLDivElement>
>(({ className, ...props }, ref) => {
  return (
    <DialogBackdrop
      ref={ref}
      className={cn(
        "fixed inset-0 z-50 bg-black/50",
        className
      )}
      {...props}
    />
  );
});

DialogOverlay.displayName = "DialogOverlay";

interface DialogContentProps extends React.HTMLAttributes<HTMLDivElement> {
  onInteractOutside?: (e: Event) => void;
}

function DialogContent({
  className,
  children,
  onInteractOutside,
  ...props
}: DialogContentProps) {
  const { onOpenChange } = React.useContext(DialogContext);

  return (
    <>
      <DialogOverlay />
      <div className="fixed inset-0 z-50 flex items-center justify-center p-1 pointer-events-none">
        <DialogPanel
          className={cn(
            "w-full max-w-[calc(100%-2rem)] gap-4 rounded-lg border bg-background p-6 shadow-lg pointer-events-auto sm:max-w-sm",
            className
          )}
          autoFocus={false}
          {...props}
        >
          {children}
          <button
            type="button"
            onClick={() => onOpenChange?.(false)}
            className="ring-offset-background focus:ring-ring absolute top-4 right-4 rounded-xs opacity-70 transition-opacity hover:opacity-100 focus:ring-2 focus:ring-offset-2 focus:outline-hidden disabled:pointer-events-none [&_svg]:pointer-events-none [&_svg]:shrink-0 [&_svg:not([class*='size-'])]:size-4"
          >
            <XIcon />
            <span className="sr-only">Close</span>
          </button>
        </DialogPanel>
      </div>
    </>
  );
}

function DialogHeader({ className, ...props }: React.HTMLAttributes<HTMLDivElement>) {
  return (
    <div
      className={cn("flex flex-col gap-2 text-center sm:text-left", className)}
      {...props}
    />
  );
}

function DialogFooter({ className, ...props }: React.HTMLAttributes<HTMLDivElement>) {
  return (
    <div
      className={cn(
        "flex flex-col-reverse gap-2 sm:flex-row sm:justify-end",
        className
      )}
      {...props}
    />
  );
}

function DialogTitleComponent({
  className,
  ...props
}: React.HTMLAttributes<HTMLHeadingElement>) {
  return (
    <DialogTitle
      className={cn("text-lg leading-none font-semibold", className)}
      {...props}
    />
  );
}

function DialogDescriptionComponent({
  className,
  ...props
}: React.HTMLAttributes<HTMLParagraphElement>) {
  return (
    <DialogDescription
      className={cn("text-muted-foreground text-sm", className)}
      {...props}
    />
  );
}

export {
  Dialog,
  DialogClose,
  DialogContent,
  DialogDescriptionComponent as DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogOverlay,
  DialogPortal,
  DialogTitleComponent as DialogTitle,
  DialogTrigger,
};
