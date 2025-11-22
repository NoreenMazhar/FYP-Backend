import { useState, useEffect } from "react";
import { Card } from "../ui/card";
import { Button } from "../ui/button";
import { Badge } from "../ui/badge";
import { Input } from "../ui/input";
import { Label } from "../ui/label";
import { Textarea } from "../ui/textarea";
import React from "react";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "../ui/dialog";
import {
  Calendar,
  FileText,
  Mail,
  Clock,
  Trash2,
  Eye,
  Loader2,
  X,
} from "lucide-react";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "../ui/table";
import {
  getReports,
  generateReport,
  getReportDetails,
  deleteReport,
  sendReportEmail,
  Report,
} from "../../utils/api";
import { Skeleton } from "../ui/skeleton";
import { toast } from "sonner";
import { useDataCache } from "../../contexts/DataCacheContext";
import jsPDF from "jspdf";
import { Download } from "lucide-react";
import html2canvas from "html2canvas";
import { VisualizationRenderer } from "./VisualizationRenderer";
import ReactDOM from "react-dom/client";

export function ReportsView() {
  const { getCachedData, setCachedData, refreshTrigger } = useDataCache();
  const [reports, setReports] = useState<Report[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [isDialogOpen, setIsDialogOpen] = useState(false);
  const [isGenerating, setIsGenerating] = useState(false);
  const [selectedReport, setSelectedReport] = useState<Report | null>(null);
  const [reportData, setReportData] = useState<any>(null);
  const [isViewDialogOpen, setIsViewDialogOpen] = useState(false);
  const [pdfBlobUrl, setPdfBlobUrl] = useState<string | null>(null);
  const [isGeneratingPdf, setIsGeneratingPdf] = useState(false);
  const [isEmailDialogOpen, setIsEmailDialogOpen] = useState(false);
  const [emailReportId, setEmailReportId] = useState<number | null>(null);
  const [isSendingEmail, setIsSendingEmail] = useState(false);
  const [emailData, setEmailData] = useState({
    to_email: "",
  });

  const [formData, setFormData] = useState({
    start_date: "",
    end_date: "",
    title: "",
    description: "",
  });

  const fetchReports = async (forceRefresh = false) => {
    const cacheKey = "reports-data";

    // Check cache first unless forcing refresh
    if (!forceRefresh && refreshTrigger === 0) {
      const cached = getCachedData<Report[]>(cacheKey);
      if (cached) {
        setReports(cached);
        setIsLoading(false);
        return;
      }
    }

    try {
      setIsLoading(true);
      setError(null);
      const response = await getReports({ limit: 50 });
      setReports(response.reports);
      setCachedData(cacheKey, response.reports);
    } catch (err: any) {
      setError(err.message || "Failed to fetch reports");
      console.error("Error fetching reports:", err);
      toast.error("Failed to load reports");
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchReports(refreshTrigger > 0);
  }, [refreshTrigger, getCachedData, setCachedData]);

  // Cleanup PDF blob URL on unmount
  useEffect(() => {
    return () => {
      if (pdfBlobUrl) {
        URL.revokeObjectURL(pdfBlobUrl);
      }
    };
  }, [pdfBlobUrl]);

  // Initialize form data when dialog opens
  useEffect(() => {
    if (isDialogOpen) {
      const today = new Date();
      const lastWeek = new Date(today);
      lastWeek.setDate(today.getDate() - 7);
      setFormData({
        start_date: lastWeek.toISOString().split("T")[0],
        end_date: today.toISOString().split("T")[0],
        title: "",
        description: "",
      });
    } else {
      // Reset form when dialog closes
      setFormData({
        start_date: "",
        end_date: "",
        title: "",
        description: "",
      });
    }
  }, [isDialogOpen]);

  // Debug: Track emailReportId changes
  useEffect(() => {
    console.log("emailReportId changed:", emailReportId);
  }, [emailReportId]);

  // Debug: Track email dialog state
  useEffect(() => {
    console.log(
      "isEmailDialogOpen changed:",
      isEmailDialogOpen,
      "emailReportId:",
      emailReportId
    );
  }, [isEmailDialogOpen, emailReportId]);

  const handleGenerateReport = async () => {
    if (!formData.start_date || !formData.end_date) {
      toast.error("Please select start and end dates");
      return;
    }

    if (!formData.title || formData.title.trim() === "") {
      toast.error("Please provide a report title");
      return;
    }

    if (!formData.description || formData.description.trim() === "") {
      toast.error("Please provide a report description");
      return;
    }

    try {
      setIsGenerating(true);
      await generateReport({
        start_date: formData.start_date,
        end_date: formData.end_date,
        title: formData.title.trim(),
        description: formData.description.trim(),
      });
      toast.success("Report generated successfully");
      setIsDialogOpen(false);
      setFormData({
        start_date: "",
        end_date: "",
        title: "",
        description: "",
      });
      fetchReports();
    } catch (err: any) {
      toast.error(err.message || "Failed to generate report");
    } finally {
      setIsGenerating(false);
    }
  };

  // Helper function to capture chart as image
  const captureChartAsImage = async (
    vizConfig: any,
    title: string
  ): Promise<string | null> => {
    return new Promise((resolve) => {
      // Create a temporary container
      const tempContainer = document.createElement("div");
      tempContainer.style.position = "fixed";
      tempContainer.style.left = "-10000px";
      tempContainer.style.top = "0";
      tempContainer.style.width = "800px";
      tempContainer.style.height = "400px";
      tempContainer.style.backgroundColor = "#ffffff";
      tempContainer.style.padding = "20px";
      tempContainer.style.boxSizing = "border-box";
      document.body.appendChild(tempContainer);

      // Create a root and render the chart
      const root = ReactDOM.createRoot(tempContainer);
      root.render(
        React.createElement(VisualizationRenderer, {
          config: vizConfig,
          title: title,
          noCard: true,
        })
      );

      // Wait for chart to render, then capture
      // Use multiple timeouts to ensure chart is fully rendered
      setTimeout(async () => {
        try {
          // Wait a bit more for SVG rendering
          await new Promise((r) => setTimeout(r, 500));

          const canvas = await html2canvas(tempContainer, {
            backgroundColor: "#ffffff",
            scale: 2,
            logging: false,
            useCORS: true,
            allowTaint: true,
          });
          const imgData = canvas.toDataURL("image/png", 0.95);

          // Cleanup
          root.unmount();
          setTimeout(() => {
            if (tempContainer.parentNode) {
              document.body.removeChild(tempContainer);
            }
          }, 100);

          resolve(imgData);
        } catch (error) {
          console.error("Error capturing chart:", error);
          try {
            root.unmount();
            if (tempContainer.parentNode) {
              document.body.removeChild(tempContainer);
            }
          } catch (cleanupError) {
            console.error("Error during cleanup:", cleanupError);
          }
          resolve(null);
        }
      }, 1500); // Wait 1.5 seconds for chart to render
    });
  };

  const generatePDF = async (report: Report, data: any) => {
    setIsGeneratingPdf(true);
    try {
      const pdf = new jsPDF("p", "mm", "a4");
      const pageWidth = pdf.internal.pageSize.getWidth();
      const pageHeight = pdf.internal.pageSize.getHeight();
      let yPosition = 20;
      const margin = 20;
      const lineHeight = 7;
      const maxWidth = pageWidth - 2 * margin;
      const chartHeight = 60; // Height in mm for charts

      // Helper function to add new page if needed
      const checkNewPage = (requiredHeight: number) => {
        if (yPosition + requiredHeight > pageHeight - margin) {
          pdf.addPage();
          yPosition = 20;
          return true;
        }
        return false;
      };

      // Title
      pdf.setFontSize(20);
      pdf.setFont("helvetica", "bold");
      pdf.text(report.title || `Report #${report.id}`, margin, yPosition);
      yPosition += 10;

      // Description
      if (report.description) {
        pdf.setFontSize(12);
        pdf.setFont("helvetica", "normal");
        const descLines = pdf.splitTextToSize(report.description, maxWidth);
        descLines.forEach((line: string) => {
          checkNewPage(lineHeight);
          pdf.text(line, margin, yPosition);
          yPosition += lineHeight;
        });
        yPosition += 5;
      }

      // Metadata
      pdf.setFontSize(10);
      pdf.setFont("helvetica", "italic");
      pdf.text(`Created: ${formatDate(report.created_at)}`, margin, yPosition);
      yPosition += lineHeight;

      if (data.report_data?.start_date && data.report_data?.end_date) {
        pdf.text(
          `Period: ${formatDate(data.report_data.start_date)} - ${formatDate(
            data.report_data.end_date
          )}`,
          margin,
          yPosition
        );
        yPosition += lineHeight;
      }

      yPosition += 5;

      // Summary Section
      if (data.report_data?.summary) {
        checkNewPage(15);
        pdf.setFontSize(16);
        pdf.setFont("helvetica", "bold");
        pdf.text("Summary", margin, yPosition);
        yPosition += 10;

        pdf.setFontSize(11);
        pdf.setFont("helvetica", "normal");

        // Overview
        if (data.report_data.summary.overview) {
          const overviewLines = pdf.splitTextToSize(
            data.report_data.summary.overview,
            maxWidth
          );
          overviewLines.forEach((line: string) => {
            checkNewPage(lineHeight);
            pdf.text(line, margin, yPosition);
            yPosition += lineHeight;
          });
          yPosition += 5;
        }

        // Key Metrics
        if (data.report_data.summary.key_metrics) {
          pdf.setFontSize(12);
          pdf.setFont("helvetica", "bold");
          pdf.text("Key Metrics:", margin, yPosition);
          yPosition += 8;

          pdf.setFontSize(10);
          pdf.setFont("helvetica", "normal");
          Object.entries(data.report_data.summary.key_metrics).forEach(
            ([key, value]) => {
              checkNewPage(lineHeight);
              const label = key
                .replace(/_/g, " ")
                .replace(/\b\w/g, (l: string) => l.toUpperCase());
              const val =
                typeof value === "number"
                  ? value.toLocaleString()
                  : String(value);
              pdf.text(`${label}: ${val}`, margin + 5, yPosition);
              yPosition += lineHeight;
            }
          );
          yPosition += 5;
        }

        // System Health
        if (data.report_data.summary.system_health) {
          pdf.setFontSize(12);
          pdf.setFont("helvetica", "bold");
          pdf.text("System Health:", margin, yPosition);
          yPosition += 8;

          pdf.setFontSize(10);
          pdf.setFont("helvetica", "normal");
          Object.entries(data.report_data.summary.system_health).forEach(
            ([key, value]) => {
              checkNewPage(lineHeight);
              const label = key
                .replace(/_/g, " ")
                .replace(/\b\w/g, (l: string) => l.toUpperCase());
              pdf.text(`${label}: ${String(value)}`, margin + 5, yPosition);
              yPosition += lineHeight;
            }
          );
          yPosition += 5;
        }

        // Recommendations
        if (
          data.report_data.summary.recommendations &&
          data.report_data.summary.recommendations.length > 0
        ) {
          pdf.setFontSize(12);
          pdf.setFont("helvetica", "bold");
          pdf.text("Recommendations:", margin, yPosition);
          yPosition += 8;

          pdf.setFontSize(10);
          pdf.setFont("helvetica", "normal");
          data.report_data.summary.recommendations.forEach((rec: string) => {
            checkNewPage(lineHeight);
            pdf.text(`• ${rec}`, margin + 5, yPosition);
            yPosition += lineHeight;
          });
        }
      }

      // Sections
      if (data.report_data?.sections) {
        for (const section of data.report_data.sections) {
          checkNewPage(20);
          yPosition += 5;

          // Section Title
          pdf.setFontSize(14);
          pdf.setFont("helvetica", "bold");
          pdf.text(section.title, margin, yPosition);
          yPosition += 8;

          // Section Description
          if (section.description) {
            pdf.setFontSize(10);
            pdf.setFont("helvetica", "normal");
            const descLines = pdf.splitTextToSize(
              section.description,
              maxWidth
            );
            descLines.forEach((line: string) => {
              checkNewPage(lineHeight);
              pdf.text(line, margin, yPosition);
              yPosition += lineHeight;
            });
            yPosition += 5;
          }

          // Insights
          if (section.insights && section.insights.length > 0) {
            pdf.setFontSize(11);
            pdf.setFont("helvetica", "bold");
            pdf.text("Key Insights:", margin, yPosition);
            yPosition += 7;

            pdf.setFontSize(9);
            pdf.setFont("helvetica", "normal");
            section.insights.forEach((insight: string) => {
              checkNewPage(lineHeight);
              pdf.text(`• ${insight}`, margin + 5, yPosition);
              yPosition += lineHeight;
            });
            yPosition += 5;
          }

          // Visualization Chart
          if (section.visualization?.Data) {
            const vizData = section.visualization.Data;
            if (vizData.X && vizData.Y && vizData.X.length > 0) {
              // Transform visualization data
              const vizConfig = {
                x: vizData.X || [],
                y: vizData.Y || [],
                plot_type:
                  section.visualization?.["Plot-type"]?.toLowerCase() || "bar",
                x_axis_label: section.visualization?.["X-axis-label"],
                y_axis_label: section.visualization?.["Y-axis-label"],
                description:
                  section.visualization?.Description || section.description,
              };

              const chartTitle =
                section.visualization?.Description || section.title || "Chart";

              // Add chart title
              pdf.setFontSize(11);
              pdf.setFont("helvetica", "bold");
              pdf.text(chartTitle, margin, yPosition);
              yPosition += 7;

              // Capture chart as image
              try {
                const chartImage = await captureChartAsImage(
                  vizConfig,
                  chartTitle
                );

                if (chartImage) {
                  checkNewPage(chartHeight);
                  // Add image to PDF (scaled to fit page width)
                  const imgWidth = pageWidth - 2 * margin;
                  pdf.addImage(
                    chartImage,
                    "PNG",
                    margin,
                    yPosition,
                    imgWidth,
                    chartHeight
                  );
                  yPosition += chartHeight + 5;
                } else {
                  // Fallback to text if image capture fails
                  pdf.setFontSize(8);
                  pdf.setFont("helvetica", "normal");
                  pdf.text(
                    `Data points: ${vizData.X.length}`,
                    margin + 5,
                    yPosition
                  );
                  yPosition += lineHeight;

                  if (vizData.Y.length > 0) {
                    const maxY = Math.max(
                      ...vizData.Y.map((y: any) => Number(y))
                    );
                    const minY = Math.min(
                      ...vizData.Y.map((y: any) => Number(y))
                    );
                    pdf.text(
                      `Range: ${minY.toLocaleString()} - ${maxY.toLocaleString()}`,
                      margin + 5,
                      yPosition
                    );
                    yPosition += lineHeight;
                  }
                }
              } catch (error) {
                console.error("Error adding chart to PDF:", error);
                // Fallback to text
                pdf.setFontSize(8);
                pdf.setFont("helvetica", "normal");
                pdf.text(
                  `Data points: ${vizData.X.length}`,
                  margin + 5,
                  yPosition
                );
                yPosition += lineHeight;
              }
            }
          }
        }
      }

      // Generate blob URL
      const pdfBlob = pdf.output("blob");
      const url = URL.createObjectURL(pdfBlob);
      setPdfBlobUrl(url);
    } catch (error: any) {
      console.error("Error generating PDF:", error);
      toast.error("Failed to generate PDF");
    } finally {
      setIsGeneratingPdf(false);
    }
  };

  const handleViewReport = async (reportId: number) => {
    try {
      const details = await getReportDetails(reportId);
      setSelectedReport(details.report);
      setReportData(details);
      setIsViewDialogOpen(true);
      // Generate PDF when dialog opens
      await generatePDF(details.report, details);
    } catch (err: any) {
      toast.error(err.message || "Failed to load report details");
    }
  };

  const handleDownloadPDF = () => {
    if (pdfBlobUrl && selectedReport) {
      const link = document.createElement("a");
      link.href = pdfBlobUrl;
      link.download = `${
        selectedReport.title || `Report_${selectedReport.id}`
      }.pdf`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
    }
  };

  const handleDeleteReport = async (reportId: number) => {
    if (!confirm("Are you sure you want to delete this report?")) return;
    try {
      await deleteReport(reportId);
      toast.success("Report deleted successfully");
      fetchReports();
    } catch (err: any) {
      toast.error(err.message || "Failed to delete report");
    }
  };

  const handleSendEmail = (reportId: number) => {
    console.log("handleSendEmail called with reportId:", reportId);
    setEmailReportId(reportId);
    setEmailData({
      to_email: "",
    });
    setIsEmailDialogOpen(true);
    console.log("Email dialog opened, emailReportId should be:", reportId);
  };

  const handleSendEmailSubmit = async (e?: React.MouseEvent) => {
    console.log("=== handleSendEmailSubmit START ===");
    console.log("Event object:", e);
    e?.preventDefault();
    e?.stopPropagation();

    console.log("handleSendEmailSubmit called", {
      emailData,
      emailReportId,
      to_email: emailData.to_email,
      to_email_trimmed: emailData.to_email.trim(),
      hasEmail: !!emailData.to_email.trim(),
    });

    // Early return check with logging
    if (!emailReportId) {
      console.error("ERROR: emailReportId is null/undefined!", {
        emailReportId,
        emailData,
      });
      toast.error("Report ID is missing. Please try again.");
      return;
    }

    if (!emailData.to_email || !emailData.to_email.trim()) {
      console.log("Validation failed: No email address");
      toast.error("Please enter at least one recipient email address");
      return;
    }

    // Validate email format (basic validation)
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    const toEmails = emailData.to_email
      .split(",")
      .map((e) => e.trim())
      .filter((e) => e);

    for (const email of toEmails) {
      if (!emailRegex.test(email)) {
        toast.error(`Invalid email address: ${email}`);
        return;
      }
    }

    // This check is now above, but keeping for safety
    if (!emailReportId) {
      console.error("Validation failed: No report ID", emailReportId);
      toast.error("Report ID is missing");
      return;
    }

    console.log("All validations passed, proceeding to send email...");

    try {
      setIsSendingEmail(true);
      console.log("Sending email with data:", { toEmails, emailReportId });

      // Send email to each recipient in "To" field (each gets it as primary recipient)
      const sendPromises = toEmails.map((toEmail) => {
        console.log("Sending email to:", toEmail);
        return sendReportEmail({
          to_email: toEmail,
          report_id: emailReportId,
        });
      });

      const results = await Promise.all(sendPromises);
      const successCount = results.length;
      console.log("Email sent successfully:", results);

      toast.success(`Report sent successfully to ${successCount} recipient(s)`);

      setIsEmailDialogOpen(false);
      setEmailData({
        to_email: "",
      });
      setEmailReportId(null);
    } catch (err: any) {
      console.error("Error sending email:", err);
      toast.error(err.message || "Failed to send report email");
    } finally {
      setIsSendingEmail(false);
    }
  };

  const handleQuickGenerate = (days: number) => {
    const endDate = new Date();
    const startDate = new Date();
    startDate.setDate(endDate.getDate() - days);

    setFormData({
      start_date: startDate.toISOString().split("T")[0],
      end_date: endDate.toISOString().split("T")[0],
      title: `${
        days === 1 ? "Daily" : days === 7 ? "Weekly" : "Monthly"
      } Report - ${endDate.toLocaleDateString()}`,
      description: "",
    });
    setIsDialogOpen(true);
  };

  const formatDate = (dateString: string) => {
    try {
      return new Date(dateString).toLocaleDateString();
    } catch {
      return dateString;
    }
  };

  return (
    <div className="space-y-6 relative">
      {/* Loading Overlay for Report Generation */}
      {isGenerating && (
        <div className="fixed inset-0 bg-background/80 backdrop-blur-sm z-50 flex items-center justify-center">
          <Card className="p-8 border-border/40 bg-card/90 backdrop-blur-sm shadow-lg max-w-md w-full mx-4">
            <div className="flex flex-col items-center justify-center space-y-4">
              <Loader2 className="w-12 h-12 text-primary animate-spin" />
              <div className="text-center space-y-2">
                <h3 className="text-lg font-semibold">Generating Report</h3>
                <p className="text-sm text-muted-foreground">
                  Please wait while we generate your report. This may take a few
                  moments...
                </p>
                <div className="flex items-center justify-center gap-2 mt-4">
                  <div
                    className="w-2 h-2 bg-primary rounded-full animate-bounce"
                    style={{ animationDelay: "0ms" }}
                  ></div>
                  <div
                    className="w-2 h-2 bg-primary rounded-full animate-bounce"
                    style={{ animationDelay: "150ms" }}
                  ></div>
                  <div
                    className="w-2 h-2 bg-primary rounded-full animate-bounce"
                    style={{ animationDelay: "300ms" }}
                  ></div>
                </div>
              </div>
            </div>
          </Card>
        </div>
      )}

      {/* Loading Overlay for Email Sending */}
      {isSendingEmail && (
        <div className="fixed inset-0 bg-background/80 backdrop-blur-sm z-50 flex items-center justify-center">
          <Card className="p-8 border-border/40 bg-card/90 backdrop-blur-sm shadow-lg max-w-md w-full mx-4">
            <div className="flex flex-col items-center justify-center space-y-4">
              <Loader2 className="w-12 h-12 text-primary animate-spin" />
              <div className="text-center space-y-2">
                <h3 className="text-lg font-semibold">Sending Email</h3>
                <p className="text-sm text-muted-foreground">
                  Please wait while we send the report via email. This may take
                  a few moments...
                </p>
                <div className="flex items-center justify-center gap-2 mt-4">
                  <div
                    className="w-2 h-2 bg-primary rounded-full animate-bounce"
                    style={{ animationDelay: "0ms" }}
                  ></div>
                  <div
                    className="w-2 h-2 bg-primary rounded-full animate-bounce"
                    style={{ animationDelay: "150ms" }}
                  ></div>
                  <div
                    className="w-2 h-2 bg-primary rounded-full animate-bounce"
                    style={{ animationDelay: "300ms" }}
                  ></div>
                </div>
              </div>
            </div>
          </Card>
        </div>
      )}

      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-3xl mb-2">Reports</h2>
          <p className="text-muted-foreground">
            Generate and download comprehensive traffic reports
          </p>
        </div>
        <Dialog
          open={isDialogOpen}
          onOpenChange={(open) => {
            setIsDialogOpen(open);
            if (!open) {
              // Reset form when dialog closes
              setFormData({
                start_date: "",
                end_date: "",
                title: "",
                description: "",
              });
            }
          }}
        >
          <DialogTrigger asChild>
            <Button>
              <FileText className="w-4 h-4 mr-2" />
              Generate New Report
            </Button>
          </DialogTrigger>
          <DialogContent
            className="max-w-2xl z-[100]"
            onInteractOutside={(e: any) => isGenerating && e.preventDefault()}
          >
            <DialogHeader>
              <DialogTitle>Generate New Report</DialogTitle>
              <DialogDescription>
                Fill in the details below to generate a comprehensive traffic
                report.
              </DialogDescription>
            </DialogHeader>
            <div className="space-y-4 mt-4">
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <Label htmlFor="start_date">Start Date</Label>
                  <Input
                    id="start_date"
                    type="date"
                    value={formData.start_date}
                    onChange={(e) =>
                      setFormData({ ...formData, start_date: e.target.value })
                    }
                  />
                </div>
                <div>
                  <Label htmlFor="end_date">End Date</Label>
                  <Input
                    id="end_date"
                    type="date"
                    value={formData.end_date}
                    onChange={(e) =>
                      setFormData({ ...formData, end_date: e.target.value })
                    }
                  />
                </div>
              </div>
              <div>
                <Label htmlFor="title">
                  Report Title <span className="text-destructive">*</span>
                </Label>
                <Input
                  id="title"
                  value={formData.title}
                  onChange={(e) =>
                    setFormData({ ...formData, title: e.target.value })
                  }
                  placeholder="e.g., Weekly Traffic Report - January 2025"
                  required
                />
                <p className="text-xs text-muted-foreground mt-1">
                  Provide a descriptive title for your report
                </p>
              </div>
              <div>
                <Label htmlFor="description">
                  Report Description <span className="text-destructive">*</span>
                </Label>
                <Textarea
                  id="description"
                  value={formData.description}
                  onChange={(e) =>
                    setFormData({ ...formData, description: e.target.value })
                  }
                  placeholder="Describe the purpose and scope of this report..."
                  rows={4}
                  required
                />
                <p className="text-xs text-muted-foreground mt-1">
                  Provide a detailed description of what this report covers
                </p>
              </div>
              <Button
                onClick={handleGenerateReport}
                disabled={
                  isGenerating ||
                  !formData.start_date ||
                  !formData.end_date ||
                  !formData.title?.trim() ||
                  !formData.description?.trim()
                }
                className="w-full"
              >
                {isGenerating ? (
                  <>
                    <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                    Generating...
                  </>
                ) : (
                  "Generate Report"
                )}
              </Button>
            </div>
          </DialogContent>
        </Dialog>
      </div>

      {error && (
        <div className="p-3 bg-red-500/10 border border-red-500/20 rounded-lg text-red-400 text-sm">
          {error}
        </div>
      )}

      {/* Quick Actions */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <Card
          className="p-4 border-border/40 bg-card/60 backdrop-blur-sm hover:bg-card/80 transition-colors cursor-pointer"
          onClick={() => handleQuickGenerate(1)}
        >
          <div className="flex items-center gap-3">
            <div className="p-3 bg-blue-500/10 rounded-lg">
              <Calendar className="w-5 h-5 text-blue-500" />
            </div>
            <div>
              <p className="text-sm text-muted-foreground">Daily Report</p>
              <p className="text-sm">Quick Generate</p>
            </div>
          </div>
        </Card>
        <Card
          className="p-4 border-border/40 bg-card/60 backdrop-blur-sm hover:bg-card/80 transition-colors cursor-pointer"
          onClick={() => handleQuickGenerate(7)}
        >
          <div className="flex items-center gap-3">
            <div className="p-3 bg-green-500/10 rounded-lg">
              <FileText className="w-5 h-5 text-green-500" />
            </div>
            <div>
              <p className="text-sm text-muted-foreground">Weekly Report</p>
              <p className="text-sm">Quick Generate</p>
            </div>
          </div>
        </Card>
        <Card className="p-4 border-border/40 bg-card/60 backdrop-blur-sm hover:bg-card/80 transition-colors cursor-pointer">
          <div className="flex items-center gap-3">
            <div className="p-3 bg-purple-500/10 rounded-lg">
              <Mail className="w-5 h-5 text-purple-500" />
            </div>
            <div>
              <p className="text-sm text-muted-foreground">Email Report</p>
              <p className="text-sm">Schedule Send</p>
            </div>
          </div>
        </Card>
        <Card className="p-4 border-border/40 bg-card/60 backdrop-blur-sm hover:bg-card/80 transition-colors cursor-pointer">
          <div className="flex items-center gap-3">
            <div className="p-3 bg-orange-500/10 rounded-lg">
              <Clock className="w-5 h-5 text-orange-500" />
            </div>
            <div>
              <p className="text-sm text-muted-foreground">Auto Reports</p>
              <p className="text-sm">Configure</p>
            </div>
          </div>
        </Card>
      </div>

      {/* Reports Table */}
      <Card className="p-6 border-border/40 bg-card/60 backdrop-blur-sm">
        <div className="mb-4">
          <h3>Recent Reports</h3>
          <p className="text-muted-foreground">
            View and download generated reports
          </p>
        </div>
        {isLoading ? (
          <div className="space-y-2">
            {[1, 2, 3].map((i) => (
              <Skeleton key={i} className="h-12 w-full" />
            ))}
          </div>
        ) : reports.length === 0 ? (
          <div className="text-center py-8 text-muted-foreground">
            <FileText className="w-12 h-12 mx-auto mb-4 opacity-50" />
            <p>No reports found. Generate your first report to get started.</p>
          </div>
        ) : (
          <div className="rounded-md border border-border/40">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Report Name</TableHead>
                  <TableHead>Date</TableHead>
                  <TableHead>Status</TableHead>
                  <TableHead className="text-right">Actions</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {reports.map((report) => (
                  <TableRow key={report.id}>
                    <TableCell>
                      <div className="flex items-center gap-2">
                        <FileText className="w-4 h-4 text-blue-500" />
                        {report.title || `Report #${report.id}`}
                      </div>
                    </TableCell>
                    <TableCell className="text-muted-foreground">
                      {formatDate(report.created_at)}
                    </TableCell>
                    <TableCell>
                      <Badge
                        variant={
                          report.status === "completed"
                            ? "default"
                            : "secondary"
                        }
                        className={
                          report.status === "completed"
                            ? "bg-green-500/90"
                            : "bg-orange-500/90"
                        }
                      >
                        {report.status}
                      </Badge>
                    </TableCell>
                    <TableCell className="text-right">
                      <div className="flex justify-end gap-2">
                        <Button
                          variant="ghost"
                          size="icon"
                          onClick={() => handleViewReport(report.id)}
                          title="View Report"
                        >
                          <Eye className="w-4 h-4" />
                        </Button>
                        <Button
                          variant="ghost"
                          size="icon"
                          onClick={() => handleSendEmail(report.id)}
                          title="Send Email"
                          className="text-blue-500 hover:text-blue-600"
                        >
                          <Mail className="w-4 h-4" />
                        </Button>
                        <Button
                          variant="ghost"
                          size="icon"
                          onClick={() => handleDeleteReport(report.id)}
                          className="text-red-500 hover:text-red-600"
                          title="Delete Report"
                        >
                          <Trash2 className="w-4 h-4" />
                        </Button>
                      </div>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </div>
        )}
      </Card>

      {/* Report Details Dialog */}
      <Dialog
        open={isViewDialogOpen}
        onOpenChange={(open: any) => {
          // This handler is kept for compatibility but won't be called
          // due to preventAutoClose prop
          if (open === false) {
            setIsViewDialogOpen(false);
            setSelectedReport(null);
            setReportData(null);
            if (pdfBlobUrl) {
              URL.revokeObjectURL(pdfBlobUrl);
              setPdfBlobUrl(null);
            }
          }
        }}
        preventAutoClose={true}
      >
        <DialogContent
          className="!w-[98vw] !max-w-[98vw] sm:!max-w-[98vw] md:!max-w-[98vw] lg:!max-w-[98vw] !h-[95vh] !max-h-[95vh] z-[100] !p-0 flex flex-col [&>button]:hidden"
          onInteractOutside={(e) => {
            // Prevent closing on outside clicks - user must use close button
            e.preventDefault();
            e.stopPropagation();
          }}
          onKeyDown={(e) => {
            // Prevent Escape key from closing the dialog
            if (e.key === "Escape") {
              e.preventDefault();
              e.stopPropagation();
            }
          }}
        >
          {selectedReport && reportData ? (
            <>
              <DialogHeader className="flex-shrink-0 px-6 pt-6 pb-4 border-b border-border/40 flex flex-row items-center justify-between relative z-50 bg-background">
                <div className="flex-1">
                  <DialogTitle className="text-2xl">
                    {selectedReport.title || `Report #${selectedReport.id}`}
                  </DialogTitle>
                  <DialogDescription className="text-base mt-2">
                    {selectedReport.description ||
                      "View detailed information and data for this report."}
                  </DialogDescription>
                </div>
                <div
                  className="flex items-center gap-2 ml-4 relative z-[60]"
                  onClick={(e) => e.stopPropagation()}
                  onMouseDown={(e) => e.stopPropagation()}
                >
                  {pdfBlobUrl && (
                    <Button
                      type="button"
                      onClick={(e) => {
                        e.preventDefault();
                        e.stopPropagation();
                        console.log("Download button clicked");
                        handleDownloadPDF();
                      }}
                      onMouseDown={(e) => {
                        e.stopPropagation();
                        e.preventDefault();
                      }}
                      variant="outline"
                      size="sm"
                      className="relative z-[70] pointer-events-auto cursor-pointer"
                      style={{ pointerEvents: "auto", zIndex: 70 }}
                    >
                      <Download className="w-4 h-4 mr-2" />
                      Download PDF
                    </Button>
                  )}
                  <Button
                    type="button"
                    variant="ghost"
                    size="icon"
                    onClick={(e) => {
                      e.preventDefault();
                      e.stopPropagation();
                      console.log("Close button clicked");
                      // Explicitly close the dialog and clean up
                      setIsViewDialogOpen(false);
                      setSelectedReport(null);
                      setReportData(null);
                      // Clean up PDF blob URL
                      if (pdfBlobUrl) {
                        URL.revokeObjectURL(pdfBlobUrl);
                        setPdfBlobUrl(null);
                      }
                    }}
                    onMouseDown={(e) => {
                      e.stopPropagation();
                      e.preventDefault();
                    }}
                    className="h-8 w-8 relative z-[70] pointer-events-auto cursor-pointer"
                    style={{ pointerEvents: "auto", zIndex: 70 }}
                  >
                    <X className="w-4 h-4" />
                  </Button>
                </div>
              </DialogHeader>
              <div className="flex-1 overflow-hidden">
                {isGeneratingPdf ? (
                  <div className="flex items-center justify-center h-full">
                    <div className="text-center">
                      <Loader2 className="w-8 h-8 animate-spin mx-auto mb-4" />
                      <p className="text-muted-foreground">Generating PDF...</p>
                    </div>
                  </div>
                ) : pdfBlobUrl ? (
                  <iframe
                    src={pdfBlobUrl}
                    className="w-full h-full border-0"
                    title="Report PDF"
                    style={{ pointerEvents: "auto" }}
                  />
                ) : (
                  <div className="flex items-center justify-center h-full">
                    <p className="text-muted-foreground">Loading PDF...</p>
                  </div>
                )}
              </div>
            </>
          ) : (
            <div className="py-8 text-center text-muted-foreground">
              <Loader2 className="w-8 h-8 animate-spin mx-auto mb-4" />
              <p>Loading report details...</p>
            </div>
          )}
        </DialogContent>
      </Dialog>

      {/* Send Email Dialog */}
      <Dialog
        open={isEmailDialogOpen}
        onOpenChange={(open: any) => {
          setIsEmailDialogOpen(open);
          if (!open) {
            setEmailData({ to_email: "" });
            setEmailReportId(null);
          }
        }}
      >
        <DialogContent
          className="!max-w-sm z-[100]"
          onInteractOutside={(e) => {
            // Prevent closing on outside clicks when sending email
            if (isSendingEmail) {
              e.preventDefault();
            }
          }}
        >
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2">
              <Mail className="w-5 h-5" />
              Send Report via Email
            </DialogTitle>
            <DialogDescription>
              Enter recipient email addresses to send this report.
            </DialogDescription>
          </DialogHeader>
          <div className="space-y-4 mt-4">
            <div>
              <Label htmlFor="to_email">
                To <span className="text-destructive">*</span>
              </Label>
              <Input
                id="to_email"
                type="text"
                value={emailData.to_email}
                onChange={(e) => setEmailData({ to_email: e.target.value })}
                onKeyDown={(e) => {
                  e.stopPropagation();
                  // Handle Enter key to submit
                  if (
                    e.key === "Enter" &&
                    !isSendingEmail &&
                    emailData.to_email.trim()
                  ) {
                    e.preventDefault();
                    console.log("Enter key pressed, triggering submit");
                    handleSendEmailSubmit();
                  }
                }}
                onClick={(e) => e.stopPropagation()}
                placeholder="email@example.com, another@example.com"
                disabled={isSendingEmail}
                autoFocus
              />
              <p className="text-xs text-muted-foreground mt-1">
                Enter email addresses separated by commas for multiple
                recipients
              </p>
            </div>
            <div className="flex gap-2 pt-2">
              <Button
                type="button"
                onClick={(e) => {
                  console.log("=== Send Email button clicked! ===", {
                    emailData,
                    emailReportId,
                    isSendingEmail,
                    isDisabled: isSendingEmail || !emailData.to_email.trim(),
                    eventType: e.type,
                    currentTarget: e.currentTarget,
                  });
                  e.preventDefault();
                  e.stopPropagation();
                  console.log("About to call handleSendEmailSubmit...");
                  handleSendEmailSubmit(e);
                  console.log("handleSendEmailSubmit call completed (async)");
                }}
                onMouseDown={(e) => {
                  // Don't prevent default on mousedown, just stop propagation
                  e.stopPropagation();
                }}
                disabled={isSendingEmail || !emailData.to_email.trim()}
                className="flex-1"
              >
                {isSendingEmail ? (
                  <>
                    <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                    Sending...
                  </>
                ) : (
                  <>
                    <Mail className="w-4 h-4 mr-2" />
                    Send Email
                  </>
                )}
              </Button>
              <Button
                variant="outline"
                onClick={() => {
                  setIsEmailDialogOpen(false);
                  setEmailData({ to_email: "" });
                  setEmailReportId(null);
                }}
                disabled={isSendingEmail}
              >
                Cancel
              </Button>
            </div>
          </div>
        </DialogContent>
      </Dialog>
    </div>
  );
}
