import React, { useState } from "react";
import { Card } from "../ui/card";
import { Button } from "../ui/button";
import { Textarea } from "../ui/textarea";
import {
  Sparkles,
  Send,
  Copy,
  Database,
  ChevronDown,
  ChevronUp,
  Code,
  FileText,
} from "lucide-react";
import { Badge } from "../ui/badge";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "../ui/collapsible";
import { Separator } from "../ui/separator";
import { queryAgent } from "../../utils/api";
import { toast } from "sonner";

export function AIQueryInterface() {
  const [query, setQuery] = useState("");
  const [isLoading, setIsLoading] = useState(false);

  // new state for results
  const [result, setResult] = useState<any | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [sqlView, setSqlView] = useState<string | null>(null);
  const [showSQL, setShowSQL] = useState(false);
  const [showRawResponse, setShowRawResponse] = useState(false);

  const exampleQueries = [
    "SProvide and analysis fo rhte liscence plate Z88TOIY",
    "Find entries with OCR score below 0.7",
    "Count vehicles by type for the year 2020 to 2025",
  ];

  const handleSubmit = async () => {
    if (!query.trim()) return;
    setIsLoading(true);
    setError(null);
    setResult(null);
    setSqlView(null);

    try {
      const res = await queryAgent(query);
      setResult(res);
      // Extract SQL if available in response
      if (res.executed_sql) {
        setSqlView(res.executed_sql);
      }
      setShowSQL(false);
      setShowRawResponse(false);
    } catch (e: any) {
      setError(e?.message ?? "Unknown error");
      toast.error("Query failed", { description: e?.message });
    } finally {
      setIsLoading(false);
    }
  };

  const handleCopySQL = () => {
    if (sqlView) {
      navigator.clipboard.writeText(sqlView);
      toast.success("SQL copied to clipboard");
    }
  };

  const parseMarkdownResponse = (text: string) => {
    if (!text) return [];

    type SectionState = {
      type: string;
      content: string[];
      level?: number;
    };

    const sections: { type: string; content: string; level?: number }[] = [];
    const lines = text.split("\n");
    let currentSection: SectionState | null = null;

    lines.forEach((line) => {
      // Headers
      if (line.startsWith("#")) {
        if (currentSection) {
          sections.push({
            type: currentSection.type,
            content: currentSection.content.join("\n"),
            level: currentSection.level,
          });
        }
        const match = line.match(/^#+/);
        const level = match ? match[0].length : 1;
        const content = line.replace(/^#+\s*/, "");
        currentSection = { type: "header", content: [content], level };
      }
      // Bullet points
      else if (line.trim().startsWith("- ")) {
        if (!currentSection || currentSection.type !== "list") {
          if (currentSection) {
            sections.push({
              type: currentSection.type,
              content: currentSection.content.join("\n"),
              level: currentSection.level,
            });
          }
          currentSection = { type: "list", content: [] };
        }
        currentSection.content.push(line.trim().substring(2));
      }
      // Regular text
      else if (line.trim()) {
        if (!currentSection || currentSection.type === "header") {
          if (currentSection && currentSection.type === "header") {
            sections.push({
              type: currentSection.type,
              content: currentSection.content.join("\n"),
              level: currentSection.level,
            });
          }
          currentSection = { type: "text", content: [] };
        }
        currentSection.content.push(line.trim());
      }
    });

    if (currentSection !== null) {
      const sectionToAdd: SectionState = currentSection;
      sections.push({
        type: sectionToAdd.type,
        content: sectionToAdd.content.join("\n"),
        level: sectionToAdd.level,
      });
    }

    return sections;
  };

  return (
    <Card className="p-6 bg-gradient-to-br from-primary/10 via-primary/5 to-accent/10 border-2 border-primary/20 backdrop-blur-sm">
      <div className="flex items-center gap-2 mb-4">
        <Sparkles className="w-5 h-5 text-primary" />
        <h3 className="text-foreground">AI Query Assistant</h3>
      </div>
      <p className="text-muted-foreground mb-4">
        Ask questions in plain English - I'll convert them to SQL and visualize
        the results
      </p>

      <div className="space-y-3 mb-4">
        <div className="flex items-start gap-2">
          <Database className="w-5 h-5 text-muted-foreground mt-1 flex-shrink-0" />
          <Textarea
            placeholder="e.g., Show me all trucks detected today with confidence score above 90%"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            className="min-h-[100px] resize-none"
          />
        </div>
        <div className="flex gap-2">
          <Button
            onClick={handleSubmit}
            disabled={!query.trim() || isLoading}
            className="bg-primary hover:bg-primary/90 shadow-lg shadow-primary/20"
          >
            {isLoading ? (
              <>
                <Sparkles className="w-4 h-4 mr-2 animate-pulse" />
                Processing...
              </>
            ) : (
              <>
                <Send className="w-4 h-4 mr-2" />
                Execute Query
              </>
            )}
          </Button>
        </div>
        {error && (
          <div className="mt-4 p-4 bg-destructive/10 border border-destructive/20 rounded-lg">
            <p className="text-destructive font-medium">Error</p>
            <p className="text-sm text-destructive/80 mt-1">{error}</p>
          </div>
        )}

        {result && (
          <div className="mt-4 space-y-4">
            {/* Main Response Card */}
            <Card className="p-6 border-border/40 bg-card/60 backdrop-blur-sm">
              <div className="flex items-center justify-between mb-4">
                <div className="flex items-center gap-2">
                  <FileText className="w-5 h-5 text-primary" />
                  <h4 className="font-semibold">Query Response</h4>
                </div>
                <Badge variant="outline" className="text-xs">
                  {result.question || "AI Response"}
                </Badge>
              </div>

              {/* Display key-value pairs from result.result */}
              {result.result &&
                typeof result.result === "object" &&
                !Array.isArray(result.result) && (
                  <div className="space-y-6">
                    {Object.entries(result.result).map(([key, value]) => {
                      // Skip keys that are not meant to be displayed
                      if (
                        key === "raw_llm_output" ||
                        (typeof value === "boolean" && !value)
                      ) {
                        return null;
                      }

                      // Convert key to a readable heading
                      const heading = key
                        .split("_")
                        .map(
                          (word) => word.charAt(0).toUpperCase() + word.slice(1)
                        )
                        .join(" ");

                      // Handle different value types
                      let content: string = "";
                      if (typeof value === "string") {
                        content = value;
                      } else if (typeof value === "object" && value !== null) {
                        // For objects, show as JSON only if it's not a simple structure
                        if (
                          Array.isArray(value) &&
                          value.length > 0 &&
                          typeof value[0] === "object"
                        ) {
                          content = JSON.stringify(value, null, 2);
                        } else {
                          content = JSON.stringify(value, null, 2);
                        }
                      } else {
                        content = String(value);
                      }

                      // Skip empty content
                      if (!content || content.trim() === "") {
                        return null;
                      }

                      // Parse markdown content
                      const sections = parseMarkdownResponse(content);

                      return (
                        <div key={key} className="space-y-3">
                          {/* Heading */}
                          <h3 className="text-lg font-semibold text-foreground border-b border-border pb-2">
                            {heading}
                          </h3>

                          {/* Content */}
                          <div className="prose prose-sm dark:prose-invert max-w-none">
                            {sections.length > 0 ? (
                              sections.map((section, idx) => {
                                if (section.type === "header") {
                                  const level = Math.min(section.level || 2, 6);
                                  const HeaderTag = `h${level}` as
                                    | "h1"
                                    | "h2"
                                    | "h3"
                                    | "h4"
                                    | "h5"
                                    | "h6";
                                  return React.createElement(
                                    HeaderTag,
                                    {
                                      key: idx,
                                      className: `font-semibold mt-4 mb-2 ${
                                        section.level === 1
                                          ? "text-xl"
                                          : section.level === 2
                                          ? "text-lg"
                                          : "text-base"
                                      }`,
                                    },
                                    section.content
                                  );
                                } else if (section.type === "list") {
                                  return (
                                    <ul
                                      key={idx}
                                      className="list-disc list-inside space-y-1 my-2 ml-4"
                                    >
                                      {section.content
                                        .split("\n")
                                        .map((item, i) => (
                                          <li
                                            key={i}
                                            className="text-sm text-muted-foreground"
                                          >
                                            {item}
                                          </li>
                                        ))}
                                    </ul>
                                  );
                                } else {
                                  return (
                                    <p
                                      key={idx}
                                      className="text-sm text-muted-foreground mb-3 leading-relaxed whitespace-pre-wrap"
                                    >
                                      {section.content}
                                    </p>
                                  );
                                }
                              })
                            ) : (
                              <p className="text-sm text-muted-foreground leading-relaxed whitespace-pre-wrap">
                                {content}
                              </p>
                            )}
                          </div>
                        </div>
                      );
                    })}
                  </div>
                )}

              {/* Fallback: If result.result is a string or doesn't have the expected structure */}
              {result.result &&
                (typeof result.result === "string" ||
                  (typeof result.result === "object" &&
                    result.result !== null &&
                    !Object.keys(result.result).length)) && (
                  <div className="prose prose-sm dark:prose-invert max-w-none">
                    {parseMarkdownResponse(
                      typeof result.result === "string"
                        ? result.result
                        : result.result?.response || ""
                    ).map((section, idx) => {
                      if (section.type === "header") {
                        const level = Math.min(section.level || 2, 6);
                        const HeaderTag = `h${level}` as
                          | "h1"
                          | "h2"
                          | "h3"
                          | "h4"
                          | "h5"
                          | "h6";
                        return React.createElement(
                          HeaderTag,
                          {
                            key: idx,
                            className: `font-semibold mt-4 mb-2 ${
                              section.level === 1
                                ? "text-xl"
                                : section.level === 2
                                ? "text-lg"
                                : "text-base"
                            }`,
                          },
                          section.content
                        );
                      } else if (section.type === "list") {
                        return (
                          <ul
                            key={idx}
                            className="list-disc list-inside space-y-1 my-2 ml-4"
                          >
                            {section.content.split("\n").map((item, i) => (
                              <li
                                key={i}
                                className="text-sm text-muted-foreground"
                              >
                                {item}
                              </li>
                            ))}
                          </ul>
                        );
                      } else {
                        return (
                          <p
                            key={idx}
                            className="text-sm text-muted-foreground mb-3 leading-relaxed"
                          >
                            {section.content}
                          </p>
                        );
                      }
                    })}
                  </div>
                )}

              {/* Key Metrics Extraction */}
              {result.result?.response && (
                <div className="mt-6 pt-4 border-t border-border">
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                    {(() => {
                      const response = result.result.response;
                      const metrics: { label: string; value: string }[] = [];

                      // Extract detection count
                      const detectionMatch = response.match(
                        /Detection Count:\s*(\d+)/i
                      );
                      if (detectionMatch) {
                        metrics.push({
                          label: "Detection Count",
                          value: detectionMatch[1],
                        });
                      }

                      // Extract average OCR
                      const ocrMatch = response.match(
                        /Average OCR Confidence:\s*([\d.]+)/i
                      );
                      if (ocrMatch) {
                        metrics.push({
                          label: "Avg OCR Score",
                          value: parseFloat(ocrMatch[1]).toFixed(2),
                        });
                      }

                      // Extract data reliability
                      const reliabilityMatch = response.match(
                        /Data Reliability:\s*(\w+)/i
                      );
                      if (reliabilityMatch) {
                        metrics.push({
                          label: "Data Reliability",
                          value: reliabilityMatch[1],
                        });
                      }

                      return metrics.map((metric, idx) => (
                        <div key={idx} className="p-3 bg-muted/50 rounded-lg">
                          <p className="text-xs text-muted-foreground mb-1">
                            {metric.label}
                          </p>
                          <p className="text-lg font-semibold">
                            {metric.value}
                          </p>
                        </div>
                      ));
                    })()}
                  </div>
                </div>
              )}
            </Card>

            {/* SQL Query Section */}
            {sqlView && (
              <Collapsible open={showSQL} onOpenChange={setShowSQL}>
                <Card className="border-border/40 bg-card/60 backdrop-blur-sm">
                  <CollapsibleTrigger className="w-full p-4 flex items-center justify-between hover:bg-muted/50 transition-colors">
                    <div className="flex items-center gap-2">
                      <Code className="w-4 h-4 text-primary" />
                      <span className="font-medium">Executed SQL Query</span>
                    </div>
                    {showSQL ? (
                      <ChevronUp className="w-4 h-4" />
                    ) : (
                      <ChevronDown className="w-4 h-4" />
                    )}
                  </CollapsibleTrigger>
                  <CollapsibleContent>
                    <div className="px-4 pb-4">
                      <div className="relative">
                        <pre className="p-4 bg-muted rounded-lg text-xs overflow-x-auto font-mono">
                          <code>{sqlView}</code>
                        </pre>
                        <Button
                          variant="ghost"
                          size="sm"
                          className="absolute top-2 right-2"
                          onClick={handleCopySQL}
                        >
                          <Copy className="w-3 h-3 mr-1" />
                          Copy
                        </Button>
                      </div>
                    </div>
                  </CollapsibleContent>
                </Card>
              </Collapsible>
            )}

            {/* Raw Response Toggle */}
            <Collapsible
              open={showRawResponse}
              onOpenChange={setShowRawResponse}
            >
              <Card className="border-border/40 bg-card/60 backdrop-blur-sm">
                <CollapsibleTrigger className="w-full p-3 flex items-center justify-between hover:bg-muted/50 transition-colors text-sm text-muted-foreground">
                  <span>View Raw Response</span>
                  {showRawResponse ? (
                    <ChevronUp className="w-4 h-4" />
                  ) : (
                    <ChevronDown className="w-4 h-4" />
                  )}
                </CollapsibleTrigger>
                <CollapsibleContent>
                  <div className="px-4 pb-4">
                    <pre className="p-4 bg-muted rounded-lg text-xs overflow-auto font-mono">
                      {JSON.stringify(result, null, 2)}
                    </pre>
                  </div>
                </CollapsibleContent>
              </Card>
            </Collapsible>
          </div>
        )}
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
