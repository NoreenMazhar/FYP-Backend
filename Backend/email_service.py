import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from typing import List, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class EmailService:
	"""
	Email service using Gmail SMTP with credentials from environment variables.
	Uses GMAIL_USER and GMAIL_APP_PASSWORD from .env file.
	"""
	
	def __init__(self):
		self.smtp_server = "smtp.gmail.com"
		self.smtp_port = 587
		self.gmail_user = os.getenv("GMAIL_USER")
		self.gmail_app_password = os.getenv("GMAIL_APP_PASSWORD")
		
	def send_email(
			self,
			to_email: str,
			subject: str,
			body: str,
			body_html: Optional[str] = None,
			from_email: Optional[str] = None,
			cc: Optional[List[str]] = None,
			bcc: Optional[List[str]] = None,
			attachments: Optional[List[str]] = None
		) -> bool:
		"""
		Send an email using Gmail SMTP.
		
		Args:
			to_email: Recipient email address
			subject: Email subject
			body: Plain text email body
			body_html: Optional HTML email body
			from_email: Sender email (defaults to GMAIL_USER)
			cc: Optional list of CC recipients
			bcc: Optional list of BCC recipients
			attachments: Optional list of file paths to attach
		
		Returns:
			True if email sent successfully, False otherwise
		"""
		if not self.gmail_user or not self.gmail_app_password:
			logger.error("Cannot send email: Gmail credentials not configured")
			return False
		
		try:
			# Create message
			msg = MIMEMultipart('alternative')
			msg['From'] = from_email or self.gmail_user
			msg['To'] = to_email
			msg['Subject'] = subject
			
			if cc:
				msg['Cc'] = ', '.join(cc)
			
			# Add body
			msg.attach(MIMEText(body, 'plain'))
			if body_html:
				msg.attach(MIMEText(body_html, 'html'))
			
			# Add attachments
			if attachments:
				for file_path in attachments:
					if os.path.exists(file_path):
						with open(file_path, 'rb') as attachment:
							part = MIMEBase('application', 'octet-stream')
							part.set_payload(attachment.read())
							encoders.encode_base64(part)
							part.add_header(
								'Content-Disposition',
								f'attachment; filename= {os.path.basename(file_path)}'
							)
							msg.attach(part)
			
			# Determine all recipients
			recipients = [to_email]
			if cc:
				recipients.extend(cc)
			if bcc:
				recipients.extend(bcc)
			
			# Connect to SMTP server and send
			with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
				server.starttls()
				server.login(self.gmail_user, self.gmail_app_password)
				server.send_message(msg, to_addrs=recipients)
			
			logger.info(f"Email sent successfully to {to_email}")
			return True
			
		except smtplib.SMTPAuthenticationError as e:
			logger.error(f"SMTP authentication failed: {e}")
			return False
		except smtplib.SMTPException as e:
			logger.error(f"SMTP error occurred: {e}")
			return False
		except Exception as e:
			logger.exception(f"Failed to send email: {e}")
			return False
	
	def send_bulk_email(
			self,
			to_emails: List[str],
			subject: str,
			body: str,
			body_html: Optional[str] = None,
			from_email: Optional[str] = None
		) -> dict:
		"""
		Send email to multiple recipients.
		
		Args:
			to_emails: List of recipient email addresses
			subject: Email subject
			body: Plain text email body
			body_html: Optional HTML email body
			from_email: Sender email (defaults to GMAIL_USER)
		
		Returns:
			Dictionary with success count and failed emails
		"""
		results = {
			"success_count": 0,
			"failed_count": 0,
			"failed_emails": []
		}
		
		for email in to_emails:
			if self.send_email(email, subject, body, body_html, from_email):
				results["success_count"] += 1
			else:
				results["failed_count"] += 1
				results["failed_emails"].append(email)
		
		return results
	
	def format_report_as_html(self, report_data: Dict[str, Any]) -> str:
		"""
		Format a report dictionary as HTML email content.
		
		Args:
			report_data: Report data dictionary from generate_comprehensive_report
		
		Returns:
			HTML formatted email body
		"""
		html = f"""
		<!DOCTYPE html>
		<html>
		<head>
			<meta charset="UTF-8">
			<style>
				body {{
					font-family: Arial, sans-serif;
					line-height: 1.6;
					color: #333;
					max-width: 800px;
					margin: 0 auto;
					padding: 20px;
				}}
				.header {{
					background-color: #2c3e50;
					color: white;
					padding: 20px;
					border-radius: 5px;
					margin-bottom: 20px;
				}}
				.header h1 {{
					margin: 0;
					font-size: 24px;
				}}
				.metadata {{
					background-color: #ecf0f1;
					padding: 15px;
					border-radius: 5px;
					margin-bottom: 20px;
				}}
				.metadata p {{
					margin: 5px 0;
				}}
				.summary {{
					background-color: #e8f5e9;
					padding: 15px;
					border-left: 4px solid #4caf50;
					margin-bottom: 20px;
				}}
				.summary h2 {{
					margin-top: 0;
					color: #2e7d32;
				}}
				.section {{
					margin-bottom: 30px;
					padding: 15px;
					background-color: #f9f9f9;
					border-radius: 5px;
				}}
				.section h3 {{
					color: #2c3e50;
					border-bottom: 2px solid #3498db;
					padding-bottom: 10px;
				}}
				.stats-grid {{
					display: grid;
					grid-template-columns: repeat(2, 1fr);
					gap: 15px;
					margin: 20px 0;
				}}
				.stat-box {{
					background-color: white;
					padding: 15px;
					border-radius: 5px;
					box-shadow: 0 2px 4px rgba(0,0,0,0.1);
				}}
				.stat-label {{
					font-size: 12px;
					color: #7f8c8d;
					text-transform: uppercase;
				}}
				.stat-value {{
					font-size: 24px;
					font-weight: bold;
					color: #2c3e50;
					margin-top: 5px;
				}}
				.insights {{
					background-color: #fff3cd;
					padding: 10px;
					border-left: 4px solid #ffc107;
					margin-top: 10px;
				}}
				.insights ul {{
					margin: 5px 0;
					padding-left: 20px;
				}}
				.footer {{
					margin-top: 30px;
					padding-top: 20px;
					border-top: 1px solid #ddd;
					font-size: 12px;
					color: #7f8c8d;
					text-align: center;
				}}
				table {{
					width: 100%;
					border-collapse: collapse;
					margin: 15px 0;
				}}
				table th, table td {{
					padding: 10px;
					text-align: left;
					border-bottom: 1px solid #ddd;
				}}
				table th {{
					background-color: #3498db;
					color: white;
				}}
			</style>
		</head>
		<body>
			<div class="header">
				<h1>{report_data.get('title', 'Traffic Monitoring Report')}</h1>
			</div>
			
			<div class="metadata">
				<p><strong>Report Period:</strong> {report_data.get('period', {}).get('start_date', 'N/A')} to {report_data.get('period', {}).get('end_date', 'N/A')}</p>
				<p><strong>Generated At:</strong> {report_data.get('generated_at', 'N/A')}</p>
		"""
		
		# Add description if available
		description = report_data.get('description')
		if description:
			html += f'				<p><strong>Description:</strong> {description}</p>\n'
		
		html += """
			</div>
		"""
		
		# Add executive summary
		summary = report_data.get('executive_summary', {})
		if summary:
			overview = summary.get('overview', '')
			html += f"""
			<div class="summary">
				<h2>Executive Summary</h2>
				<p>{overview}</p>
			"""
			
			key_metrics = summary.get('key_metrics', {})
			if key_metrics:
				html += """
				<div class="stats-grid">
				"""
				for key, value in key_metrics.items():
					label = key.replace('_', ' ').title()
					html += f"""
					<div class="stat-box">
						<div class="stat-label">{label}</div>
						<div class="stat-value">{value}</div>
					</div>
					"""
				html += """
				</div>
				"""
			
			system_health = summary.get('system_health', {})
			if system_health:
				html += """
				<h3>System Health</h3>
				<ul>
				"""
				for key, value in system_health.items():
					label = key.replace('_', ' ').title()
					html += f"<li><strong>{label}:</strong> {value}</li>"
				html += """
				</ul>
				"""
			
			recommendations = summary.get('recommendations', [])
			if recommendations:
				html += """
				<h3>Recommendations</h3>
				<ul>
				"""
				for rec in recommendations:
					html += f"<li>{rec}</li>"
				html += """
				</ul>
				"""
			
			html += """
			</div>
			"""
		
		# Add statistics
		stats = report_data.get('statistics', {})
		if stats:
			html += """
			<div class="section">
				<h3>Key Statistics</h3>
				<div class="stats-grid">
			"""
			for key, value in stats.items():
				if value is not None:
					label = key.replace('_', ' ').title()
					html += f"""
					<div class="stat-box">
						<div class="stat-label">{label}</div>
						<div class="stat-value">{value}</div>
					</div>
					"""
			html += """
				</div>
			</div>
			"""
		
		# Add report sections
		sections = report_data.get('sections', [])
		for section in sections:
			html += f"""
			<div class="section">
				<h3>{section.get('title', 'Section')}</h3>
				<p>{section.get('description', '')}</p>
			"""
			
			insights = section.get('insights', [])
			if insights:
				html += """
				<div class="insights">
					<strong>Key Insights:</strong>
					<ul>
				"""
				for insight in insights:
					html += f"<li>{insight}</li>"
				html += """
					</ul>
				</div>
				"""
			
			visualization = section.get('visualization')
			if visualization:
				html += f"""
				<p><em>Visualization data available: {visualization.get('Description', 'Chart data included')}</em></p>
				"""
			
			html += """
			</div>
			"""
		
		# Add anomalies summary
		anomalies_summary = report_data.get('anomalies_summary', {})
		if anomalies_summary:
			html += f"""
			<div class="section">
				<h3>Anomaly Detection Summary</h3>
				<div class="stats-grid">
					<div class="stat-box">
						<div class="stat-label">Total Anomalies</div>
						<div class="stat-value">{anomalies_summary.get('total_anomalies', 0)}</div>
					</div>
					<div class="stat-box">
						<div class="stat-label">Active Anomalies</div>
						<div class="stat-value">{anomalies_summary.get('active_anomalies', 0)}</div>
					</div>
					<div class="stat-box">
						<div class="stat-label">Resolved Anomalies</div>
						<div class="stat-value">{anomalies_summary.get('resolved_anomalies', 0)}</div>
					</div>
				</div>
			</div>
			"""
		
		html += """
			<div class="footer">
				<p>This is an automated report generated by the Traffic Monitoring System.</p>
				<p>For questions or support, please contact your system administrator.</p>
			</div>
		</body>
		</html>
		"""
		
		return html
	
	def send_report_email(
			self,
			to_email: str,
			report_data: Dict[str, Any],
			from_email: Optional[str] = None,
			cc: Optional[List[str]] = None,
			bcc: Optional[List[str]] = None
		) -> bool:
		"""
		Send a report as an email with HTML formatting.
		
		Args:
			to_email: Recipient email address
			report_data: Report data dictionary from generate_comprehensive_report
			from_email: Sender email (defaults to GMAIL_USER)
			cc: Optional list of CC recipients
			bcc: Optional list of BCC recipients
		
		Returns:
			True if email sent successfully, False otherwise
		"""
		# Generate plain text version
		plain_text = f"""
		{report_data.get('title', 'Traffic Monitoring Report')}

		Report Period: {report_data.get('period', {}).get('start_date', 'N/A')} to {report_data.get('period', {}).get('end_date', 'N/A')}
		Generated At: {report_data.get('generated_at', 'N/A')}

		Executive Summary:
		{report_data.get('executive_summary', {}).get('overview', 'N/A')}

		Key Statistics:
		"""
		stats = report_data.get('statistics', {})
		for key, value in stats.items():
			if value is not None:
				plain_text += f"  {key.replace('_', ' ').title()}: {value}\n"
		
		plain_text += "\nReport sections and detailed analysis are included in the HTML version of this email.\n"
		
		# Generate HTML version
		html_content = self.format_report_as_html(report_data)
		
		# Send email
		return self.send_email(
			to_email=to_email,
			subject=report_data.get('title', 'Traffic Monitoring Report'),
			body=plain_text,
			body_html=html_content,
			from_email=from_email,
			cc=cc,
			bcc=bcc
		)


# Singleton instance
_email_service = None

def get_email_service() -> EmailService:
	"""
	Get or create the email service singleton instance.
	"""
	global _email_service
	if _email_service is None:
		_email_service = EmailService()
	return _email_service
