import { useState } from "react";
import { Card } from "../ui/card";
import { Button } from "../ui/button";
import { Textarea } from "../ui/textarea";
import { Sparkles, Send } from "lucide-react";
import { Badge } from "../ui/badge";
import { textToPlots } from "../../utils/api";
import { toast } from "sonner";
import React from "react";
interface PlotQueryInterfaceProps {
  onPlotCreated?: () => void;
}

export function PlotQueryInterface({ onPlotCreated }: PlotQueryInterfaceProps) {
  const [query, setQuery] = useState("");
  const [isLoading, setIsLoading] = useState(false);

  const exampleQueries = [
    "Show me vehicle type distribution",
    "Display hourly traffic patterns",
    "Compare inbound vs outbound traffic",
    "Show OCR score trends over time",
    "Visualize detections by device",
  ];

  const handleSubmit = async () => {
    if (!query.trim()) return;
    setIsLoading(true);

    try {
      const plots = await textToPlots({
        text_description: query,
      });

      if (plots && plots.length > 0) {
        toast.success(`Created ${plots.length} visualization(s) successfully!`);
        setQuery("");
        if (onPlotCreated) {
          onPlotCreated();
        }
      } else {
        toast.warning("No visualizations were created");
      }
    } catch (e: any) {
      toast.error("Failed to create visualization", {
        description: e?.message || "Unknown error",
      });
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <Card className="p-6 bg-gradient-to-br from-primary/10 via-primary/5 to-accent/10 border-2 border-primary/20 backdrop-blur-sm">
      <div className="flex items-center gap-2 mb-4">
        <Sparkles className="w-5 h-5 text-primary" />
        <h3 className="text-foreground">Create New Visualization</h3>
      </div>
      <p className="text-muted-foreground mb-4">
        Ask a question in plain English to generate a new chart or visualization
      </p>

      <div className="space-y-3 mb-4">
        <Textarea
          placeholder="e.g., Show me vehicle detections by hour of day"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          className="min-h-[100px] resize-none"
          onKeyDown={(e) => {
            if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
              handleSubmit();
            }
          }}
        />
        <div className="flex gap-2">
          <Button
            onClick={handleSubmit}
            disabled={!query.trim() || isLoading}
            className="bg-primary hover:bg-primary/90 shadow-lg shadow-primary/20"
          >
            {isLoading ? (
              <>
                <Sparkles className="w-4 h-4 mr-2 animate-pulse" />
                Creating...
              </>
            ) : (
              <>
                <Send className="w-4 h-4 mr-2" />
                Create Visualization
              </>
            )}
          </Button>
        </div>
      </div>

      <div>
        <p className="text-sm text-muted-foreground mb-2">
          Try these examples:
        </p>
        <div className="flex flex-wrap gap-2">
          {exampleQueries.map((example, idx) => (
            <Badge
              key={idx}
              variant="secondary"
              className="cursor-pointer hover:bg-accent transition-colors"
              onClick={() => setQuery(example)}
            >
              {example}
            </Badge>
          ))}
        </div>
      </div>
    </Card>
  );
}
