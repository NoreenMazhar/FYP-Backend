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
  Download,
  FileText,
  Mail,
  Printer,
  Clock,
  Trash2,
  Eye,
  Loader2,
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
  const [isEmailDialogOpen, setIsEmailDialogOpen] = useState(false);
  const [emailReportId, setEmailReportId] = useState<number | null>(null);
  const [isSendingEmail, setIsSendingEmail] = useState(false);
  const [emailData, setEmailData] = useState({
    to_email: "",
    cc: "",
    bcc: "",
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

  const handleViewReport = async (reportId: number) => {
    try {
      const details = await getReportDetails(reportId);
      setSelectedReport(details.report);
      setReportData(details);
      setIsViewDialogOpen(true);
    } catch (err: any) {
      toast.error(err.message || "Failed to load report details");
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
    setEmailReportId(reportId);
    setEmailData({
      to_email: "",
      cc: "",
      bcc: "",
    });
    setIsEmailDialogOpen(true);
  };

  const handleSendEmailSubmit = async () => {
    if (!emailData.to_email.trim()) {
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

    // Validate CC emails if provided
    let ccEmails: string[] = [];
    if (emailData.cc.trim()) {
      ccEmails = emailData.cc
        .split(",")
        .map((e) => e.trim())
        .filter((e) => e);
      for (const email of ccEmails) {
        if (!emailRegex.test(email)) {
          toast.error(`Invalid CC email address: ${email}`);
          return;
        }
      }
    }

    // Validate BCC emails if provided
    let bccEmails: string[] = [];
    if (emailData.bcc.trim()) {
      bccEmails = emailData.bcc
        .split(",")
        .map((e) => e.trim())
        .filter((e) => e);
      for (const email of bccEmails) {
        if (!emailRegex.test(email)) {
          toast.error(`Invalid BCC email address: ${email}`);
          return;
        }
      }
    }

    if (!emailReportId) {
      toast.error("Report ID is missing");
      return;
    }

    try {
      setIsSendingEmail(true);

      // Send email to each recipient in "To" field (each gets it as primary recipient)
      const sendPromises = toEmails.map((toEmail) =>
        sendReportEmail({
          to_email: toEmail,
          report_id: emailReportId,
          cc: ccEmails.length > 0 ? ccEmails : undefined,
          bcc: bccEmails.length > 0 ? bccEmails : undefined,
        })
      );

      const results = await Promise.all(sendPromises);
      const successCount = results.length;

      toast.success(
        `Report sent successfully to ${successCount} recipient(s)${
          ccEmails.length > 0 ? ` (CC: ${ccEmails.length})` : ""
        }${bccEmails.length > 0 ? ` (BCC: ${bccEmails.length})` : ""}`
      );

      setIsEmailDialogOpen(false);
      setEmailData({
        to_email: "",
        cc: "",
        bcc: "",
      });
      setEmailReportId(null);
    } catch (err: any) {
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
      {/* Loading Overlay */}
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
                Fill in the details below to generate a comprehensive traffic report.
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
          setIsViewDialogOpen(open);
          if (!open) {
            setSelectedReport(null);
            setReportData(null);
          }
        }}
      >
        <DialogContent className="max-w-4xl max-h-[80vh] overflow-y-auto z-[100]">
          {selectedReport ? (
            <>
              <DialogHeader>
                <DialogTitle>
                  {selectedReport.title || `Report #${selectedReport.id}`}
                </DialogTitle>
                <DialogDescription>
                  View detailed information and data for this report.
                </DialogDescription>
              </DialogHeader>
              <div className="mt-4 space-y-4">
                {selectedReport.description && (
                  <p className="text-muted-foreground">
                    {selectedReport.description}
                  </p>
                )}
                <div className="text-sm text-muted-foreground">
                  Created: {formatDate(selectedReport.created_at)}
                </div>
                {reportData?.report_data && (
                  <div className="mt-4">
                    <h4 className="font-semibold mb-2">Report Data</h4>
                    <pre className="bg-muted p-4 rounded-lg overflow-auto text-xs">
                      {JSON.stringify(reportData.report_data, null, 2)}
                    </pre>
                  </div>
                )}
                {reportData?.visualizations &&
                  reportData.visualizations.length > 0 && (
                    <div className="mt-4">
                      <h4 className="font-semibold mb-2">Visualizations</h4>
                      <p className="text-sm text-muted-foreground">
                        This report includes {reportData.visualizations.length}{" "}
                        visualization(s)
                      </p>
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
            setEmailData({ to_email: "", cc: "", bcc: "" });
            setEmailReportId(null);
          }
        }}
      >
        <DialogContent className="max-w-md z-[100]">
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
                onChange={(e) =>
                  setEmailData({ ...emailData, to_email: e.target.value })
                }
                placeholder="email@example.com, another@example.com"
                disabled={isSendingEmail}
              />
              <p className="text-xs text-muted-foreground mt-1">
                Enter email addresses separated by commas for multiple
                recipients
              </p>
            </div>
            <div>
              <Label htmlFor="cc">CC (Optional)</Label>
              <Input
                id="cc"
                type="text"
                value={emailData.cc}
                onChange={(e) =>
                  setEmailData({ ...emailData, cc: e.target.value })
                }
                placeholder="cc@example.com, cc2@example.com"
                disabled={isSendingEmail}
              />
              <p className="text-xs text-muted-foreground mt-1">
                Carbon copy recipients (comma-separated)
              </p>
            </div>
            <div>
              <Label htmlFor="bcc">BCC (Optional)</Label>
              <Input
                id="bcc"
                type="text"
                value={emailData.bcc}
                onChange={(e) =>
                  setEmailData({ ...emailData, bcc: e.target.value })
                }
                placeholder="bcc@example.com, bcc2@example.com"
                disabled={isSendingEmail}
              />
              <p className="text-xs text-muted-foreground mt-1">
                Blind carbon copy recipients (comma-separated)
              </p>
            </div>
            <div className="flex gap-2 pt-2">
              <Button
                onClick={handleSendEmailSubmit}
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
                  setEmailData({ to_email: "", cc: "", bcc: "" });
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
