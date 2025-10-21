import logging
import json
from datetime import datetime, date
from typing import Dict, List, Any, Optional
from db import Database
from plots import get_2d_plots_via_agent

logger = logging.getLogger(__name__)


def generate_comprehensive_report(
    start_date: date,
    end_date: date,
    title: Optional[str] = None,
    description: Optional[str] = None,
    created_by: Optional[int] = None,
    db: Optional[Database] = None
) -> Dict[str, Any]:
    """
    Generate a comprehensive traffic monitoring report with visualizations and summary.
    The report includes 3-4 key sections with visualizations and an executive summary.
    """
    try:
        logger.info(f"Generating report for period {start_date} to {end_date}")
        
        if not db:
            db = Database.get_instance()
        
        # Generate report data using existing visualization functions
        plots = get_2d_plots_via_agent(
            start_date=start_date,
            end_date=end_date,
            device=None,
            vehicle_type=None,
            db=db,
            created_by=created_by
        )
        
        # Get anomaly data for the period
        anomalies = db.execute("""
            SELECT anomaly_type, description, status, severity, device_id, detected_at
            FROM anomalies 
            WHERE DATE(detected_at) BETWEEN %s AND %s
            ORDER BY detected_at DESC
        """, (start_date, end_date))
        
        # Get basic statistics
        stats_query = """
        SELECT 
            COUNT(*) as total_detections,
            COUNT(DISTINCT device_name) as active_devices,
            COUNT(DISTINCT vehicle_type) as vehicle_types_detected,
            AVG(ocr_score) as avg_ocr_score,
            MIN(local_timestamp) as first_detection,
            MAX(local_timestamp) as last_detection
        FROM data_raw 
        WHERE DATE(local_timestamp) BETWEEN %s AND %s
        """
        
        stats = db.execute(stats_query, (start_date, end_date))
        stats = stats[0] if stats else {}
        
        # Generate executive summary
        summary = _generate_executive_summary(stats, anomalies, start_date, end_date)
        
        # Create report sections
        report_sections = _create_report_sections(plots, stats, anomalies)
        
        # Generate report title if not provided
        report_title = title or f"Traffic Monitoring Report - {start_date} to {end_date}"
        
        # Save report to database
        report_id = _save_report_to_db(
            db=db,
            title=report_title,
            description=description,
            created_by=created_by,
            start_date=start_date,
            end_date=end_date,
            summary=summary,
            sections=report_sections
        )
        
        # Prepare response
        report_data = {
            "report_id": report_id,
            "title": report_title,
            "description": description,
            "period": {
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat()
            },
            "generated_at": datetime.now().isoformat(),
            "executive_summary": summary,
            "sections": report_sections,
            "statistics": {
                "total_detections": stats.get('total_detections', 0),
                "active_devices": stats.get('active_devices', 0),
                "vehicle_types_detected": stats.get('vehicle_types_detected', 0),
                "average_ocr_score": round(stats.get('avg_ocr_score', 0), 2),
                "first_detection": stats.get('first_detection'),
                "last_detection": stats.get('last_detection')
            },
            "anomalies_summary": {
                "total_anomalies": len(anomalies),
                "active_anomalies": len([a for a in anomalies if a['status'] == 'active']),
                "resolved_anomalies": len([a for a in anomalies if a['status'] == 'resolved'])
            }
        }
        
        logger.info(f"Successfully generated report {report_id}")
        return report_data
        
    except Exception as exc:
        logger.exception("Failed to generate report")
        raise


def _generate_executive_summary(stats, anomalies, start_date, end_date):
    """Generate an executive summary for the report."""
    total_detections = stats.get('total_detections', 0)
    active_devices = stats.get('active_devices', 0)
    avg_ocr_score = stats.get('avg_ocr_score', 0)
    active_anomalies = len([a for a in anomalies if a['status'] == 'active'])
    
    # Calculate period duration
    period_days = (end_date - start_date).days + 1
    daily_avg = total_detections / period_days if period_days > 0 else 0
    
    summary = {
        "overview": f"Traffic monitoring report covering {period_days} days from {start_date} to {end_date}",
        "key_metrics": {
            "total_vehicle_detections": total_detections,
            "daily_average_detections": round(daily_avg, 1),
            "active_monitoring_devices": active_devices,
            "average_detection_confidence": round(avg_ocr_score, 2)
        },
        "system_health": {
            "active_anomalies": active_anomalies,
            "anomaly_status": "Normal" if active_anomalies == 0 else f"{active_anomalies} issues require attention",
            "detection_quality": "Excellent" if avg_ocr_score > 0.8 else "Good" if avg_ocr_score > 0.6 else "Needs Improvement"
        },
        "recommendations": []
    }
    
    # Add recommendations based on data
    if active_anomalies > 0:
        summary["recommendations"].append("Review and address active anomalies to improve system reliability")
    
    if avg_ocr_score < 0.7:
        summary["recommendations"].append("Consider camera calibration or maintenance to improve detection accuracy")
    
    if daily_avg < 10:
        summary["recommendations"].append("Low detection volume may indicate device issues or low traffic")
    
    return summary


def _create_report_sections(plots, stats, anomalies):
    """Create structured report sections with visualizations."""
    sections = []
    
    # Section 1: Traffic Volume Analysis
    volume_plot = next((p for p in plots if "detections per day" in p.get("Description", "").lower()), None)
    if not volume_plot:
        volume_plot = next((p for p in plots if p.get("Plot-type") == "line"), plots[0] if plots else None)
    
    sections.append({
        "title": "1. Traffic Volume Analysis",
        "description": "Overview of vehicle detection patterns and traffic flow during the reporting period",
        "visualization": volume_plot,
        "insights": [
            f"Total detections: {stats.get('total_detections', 0):,}",
            f"Daily average: {stats.get('total_detections', 0) / max(1, (stats.get('last_detection', '') and stats.get('first_detection', '')) and 1 or 1):.1f}",
            f"Peak detection period: {_get_peak_period(volume_plot) if volume_plot else 'Analysis pending'}"
        ]
    })
    
    # Section 2: Vehicle Type Distribution
    vehicle_plot = next((p for p in plots if "vehicle type" in p.get("Description", "").lower()), None)
    if not vehicle_plot:
        vehicle_plot = next((p for p in plots if p.get("Plot-type") == "bar"), plots[1] if len(plots) > 1 else None)
    
    sections.append({
        "title": "2. Vehicle Type Distribution",
        "description": "Breakdown of detected vehicles by type and classification accuracy",
        "visualization": vehicle_plot,
        "insights": [
            f"Vehicle types detected: {stats.get('vehicle_types_detected', 0)}",
            f"Average OCR confidence: {stats.get('avg_ocr_score', 0):.2f}",
            "Most common vehicle type: " + _get_most_common_vehicle_type(vehicle_plot) if vehicle_plot else "Analysis pending"
        ]
    })
    
    # Section 3: Device Performance
    device_plot = next((p for p in plots if "device" in p.get("Description", "").lower()), None)
    if not device_plot:
        device_plot = next((p for p in plots if p.get("Plot-type") == "donut"), plots[2] if len(plots) > 2 else None)
    
    sections.append({
        "title": "3. Device Performance & Coverage",
        "description": "Monitoring device status, coverage areas, and detection performance",
        "visualization": device_plot,
        "insights": [
            f"Active monitoring devices: {stats.get('active_devices', 0)}",
            "Device status: All systems operational" if stats.get('active_devices', 0) > 0 else "No active devices detected",
            "Coverage analysis: " + _get_device_coverage_analysis(device_plot) if device_plot else "Analysis pending"
        ]
    })
    
    # Section 4: Anomaly Detection & Security
    anomaly_plot = next((p for p in plots if "direction" in p.get("Description", "").lower()), None)
    if not anomaly_plot:
        anomaly_plot = next((p for p in plots if p.get("Plot-type") == "pie"), plots[3] if len(plots) > 3 else None)
    
    sections.append({
        "title": "4. Anomaly Detection & Security Analysis",
        "description": "Security monitoring, anomaly detection, and traffic pattern analysis",
        "visualization": anomaly_plot,
        "insights": [
            f"Total anomalies detected: {len(anomalies)}",
            f"Active security alerts: {len([a for a in anomalies if a['status'] == 'active'])}",
            "Security status: " + ("Secure" if len(anomalies) == 0 else "Attention required"),
            "Traffic direction analysis: " + _get_direction_analysis(anomaly_plot) if anomaly_plot else "Analysis pending"
        ]
    })
    
    return sections


def _get_peak_period(plot):
    """Extract peak period from volume plot data."""
    if not plot or not plot.get("Data"):
        return "No data available"
    
    x_data = plot["Data"].get("X", [])
    y_data = plot["Data"].get("Y", [])
    
    if not x_data or not y_data:
        return "No data available"
    
    max_idx = y_data.index(max(y_data))
    return x_data[max_idx] if max_idx < len(x_data) else "Unknown"


def _get_most_common_vehicle_type(plot):
    """Extract most common vehicle type from plot data."""
    if not plot or not plot.get("Data"):
        return "No data available"
    
    x_data = plot["Data"].get("X", [])
    y_data = plot["Data"].get("Y", [])
    
    if not x_data or not y_data:
        return "No data available"
    
    max_idx = y_data.index(max(y_data))
    return x_data[max_idx] if max_idx < len(x_data) else "Unknown"


def _get_device_coverage_analysis(plot):
    """Analyze device coverage from plot data."""
    if not plot or not plot.get("Data"):
        return "No data available"
    
    x_data = plot["Data"].get("X", [])
    y_data = plot["Data"].get("Y", [])
    
    if not x_data or not y_data:
        return "No data available"
    
    total_detections = sum(y_data)
    if total_detections == 0:
        return "No detections recorded"
    
    # Calculate coverage distribution
    coverage = []
    for i, device in enumerate(x_data):
        percentage = (y_data[i] / total_detections) * 100
        coverage.append(f"{device}: {percentage:.1f}%")
    
    return "; ".join(coverage[:3])  # Show top 3 devices


def _get_direction_analysis(plot):
    """Analyze traffic direction from plot data."""
    if not plot or not plot.get("Data"):
        return "No data available"
    
    x_data = plot["Data"].get("X", [])
    y_data = plot["Data"].get("Y", [])
    
    if not x_data or not y_data:
        return "No data available"
    
    total = sum(y_data)
    if total == 0:
        return "No direction data"
    
    # Find dominant direction
    max_idx = y_data.index(max(y_data))
    dominant_direction = x_data[max_idx]
    dominant_percentage = (y_data[max_idx] / total) * 100
    
    return f"Primary direction: {dominant_direction} ({dominant_percentage:.1f}%)"


def _save_report_to_db(db, title, description, created_by, start_date, end_date, summary, sections):
    """Save the generated report to the database."""
    try:
        # Insert report
        db.execute("""
            INSERT INTO reports (title, description, created_by, status)
            VALUES (%s, %s, %s, %s)
        """, (title, description, created_by or 1, 'published'))
        
        # Get the report ID
        report_result = db.execute("SELECT LAST_INSERT_ID() as report_id")
        report_id = report_result[0]['report_id'] if report_result else None
        
        if not report_id:
            raise Exception("Failed to get report ID")
        
        # Save report metadata
        report_metadata = {
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "summary": summary,
            "sections": sections,
            "generated_at": datetime.now().isoformat()
        }
        
        # Create a visualization entry for the report
        db.execute("""
            INSERT INTO visualizations (title, viz_type, config, created_by)
            VALUES (%s, %s, %s, %s)
        """, (
            f"Report: {title}",
            "report",
            json.dumps(report_metadata),
            created_by or 1
        ))
        
        # Link report to visualization
        viz_result = db.execute("SELECT LAST_INSERT_ID() as viz_id")
        viz_id = viz_result[0]['viz_id'] if viz_result else None
        
        if viz_id:
            db.execute("""
                INSERT INTO report_visualizations (report_id, visualization_id, position)
                VALUES (%s, %s, %s)
            """, (report_id, viz_id, 0))
        
        logger.info(f"Saved report {report_id} to database")
        return report_id
        
    except Exception as e:
        logger.error(f"Failed to save report to database: {e}")
        raise
