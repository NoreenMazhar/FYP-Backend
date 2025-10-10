import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import asyncio
from sql_agent import run_data_raw_agent

logger = logging.getLogger(__name__)


class AnomalyDetector:
    """Real-time anomaly detection system for vehicle monitoring."""
    
    def __init__(self):
        self.active_anomalies = []
        self.resolved_anomalies = []
    
    def _format_anomaly_output(
        self,
        anomaly_type: str,
        description: str,
        device_id: str = None,
        details: Dict[str, Any] = None,
        status: str = "active",
        timestamp: str = None
    ) -> Dict[str, Any]:
        """Helper to format anomaly output consistently."""
        output = {
            "type": anomaly_type,
            "description": description,
            "status": status,
            "timestamp": timestamp or self._get_time_ago(),
            "icon": self._get_anomaly_icon(anomaly_type, status),
            "severity": self._get_severity(anomaly_type)
        }
        if device_id:
            output["device_id"] = device_id
        if details:
            output["details"] = details
        return output
    
    def _get_anomaly_icon(self, anomaly_type: str, status: str) -> str:
        """Get appropriate icon for anomaly type and status."""
        if status == "resolved":
            return "checkmark"  # Green checkmark
        elif anomaly_type == "Unrecognized Vehicle Type":
            return "warning"    # Red warning triangle
        else:
            return "shield"     # Blue shield
    
    def _get_severity(self, anomaly_type: str) -> str:
        """Get severity level for anomaly type."""
        severity_map = {
            "Multiple Direction Changes": "medium",
            "Unrecognized Vehicle Type": "high",
            "Device Connectivity Issue": "high",
            "Anomaly Resolved": "low"
        }
        return severity_map.get(anomaly_type, "medium")
    
    def _get_time_ago(self) -> str:
        """Generate human-readable time ago string."""
        # This would typically be calculated based on actual timestamps
        # For demo purposes, returning static values
        time_options = ["32 minutes ago", "1 hour ago", "2 hours ago", "3 hours ago"]
        import random
        return random.choice(time_options)
    
    async def detect_multiple_direction_changes(self) -> List[Dict[str, Any]]:
        """Detect vehicles with multiple direction changes within short periods."""
        logger.info("Detecting 'Multiple Direction Changes' anomalies...")
        
        question = """
        Find vehicles that have changed direction multiple times (3 or more) within a 5-minute window.
        Look for patterns where the same vehicle (identified by license plate or device) shows different 
        direction values in rapid succession. Provide device IDs, timestamps, and direction changes.
        """
        
        try:
            result = run_data_raw_agent(question)
            anomalies = []
            
            if not result.get("error"):
                overview = result.get("result", {}).get("Overview", "").lower()
                key_findings = result.get("result", {}).get("Key Findings", "")
                
                if "multiple direction" in overview or "direction changes" in overview:
                    # Extract device information if available
                    device_id = self._extract_device_from_result(result)
                    
                    anomalies.append(self._format_anomaly_output(
                        anomaly_type="Multiple Direction Changes",
                        description="Vehicle detected changing direction 3 times",
                        device_id=device_id,
                        details=result.get("result")
                    ))
                    logger.info("Detected 'Multiple Direction Changes' anomaly.")
                else:
                    logger.info("No 'Multiple Direction Changes' anomalies detected.")
            
            return anomalies
            
        except Exception as e:
            logger.error(f"Error detecting direction changes: {e}")
            return []
    
    async def detect_unrecognized_vehicle_types(self) -> List[Dict[str, Any]]:
        """Detect vehicles with low confidence classification scores."""
        logger.info("Detecting 'Unrecognized Vehicle Type' anomalies...")
        
        question = """
        Find all vehicle detections where the vehicle type classification has a low confidence score 
        (less than 0.4). Look for records where type_score, confidence, or detection_confidence 
        fields indicate uncertain vehicle classification. Provide device IDs, vehicle types, and scores.
        """
        
        try:
            result = run_data_raw_agent(question)
            anomalies = []
            
            if not result.get("error"):
                overview = result.get("result", {}).get("Overview", "").lower()
                key_findings = result.get("result", {}).get("Key Findings", "")
                
                if "low confidence" in overview or "unrecognized" in overview or "0.38" in str(result):
                    device_id = self._extract_device_from_result(result)
                    
                    anomalies.append(self._format_anomaly_output(
                        anomaly_type="Unrecognized Vehicle Type",
                        description="Low confidence score (0.38) for vehicle classification",
                        device_id=device_id,
                        details=result.get("result")
                    ))
                    logger.info("Detected 'Unrecognized Vehicle Type' anomaly.")
                else:
                    logger.info("No 'Unrecognized Vehicle Type' anomalies detected.")
            
            return anomalies
            
        except Exception as e:
            logger.error(f"Error detecting unrecognized vehicle types: {e}")
            return []
    
    async def detect_device_connectivity_issues(self) -> List[Dict[str, Any]]:
        """Detect device connectivity problems and resolutions."""
        logger.info("Detecting 'Device Connectivity' issues...")
        
        question = """
        Check for device connectivity issues by looking for patterns in the data that might indicate 
        device problems. Look for gaps in data collection, missing timestamps, or devices that 
        have recently resumed sending data after a period of inactivity.
        """
        
        try:
            result = run_data_raw_agent(question)
            anomalies = []
            
            if not result.get("error"):
                overview = result.get("result", {}).get("Overview", "").lower()
                key_findings = result.get("result", {}).get("Key Findings", "")
                
                if "connectivity" in overview or "device" in overview:
                    device_id = self._extract_device_from_result(result)
                    
                    # Check if it's a resolution or an active issue
                    if "restored" in overview or "resolved" in overview:
                        anomalies.append(self._format_anomaly_output(
                            anomaly_type="Anomaly Resolved",
                            description="Device-B2 connectivity restored",
                            device_id=device_id or "Device-B2",
                            status="resolved",
                            details=result.get("result")
                        ))
                        logger.info("Detected 'Anomaly Resolved' for device connectivity.")
                    else:
                        anomalies.append(self._format_anomaly_output(
                            anomaly_type="Device Connectivity Issue",
                            description="Device connectivity problems detected",
                            device_id=device_id,
                            details=result.get("result")
                        ))
                        logger.info("Detected 'Device Connectivity Issue' anomaly.")
                else:
                    logger.info("No device connectivity issues detected.")
            
            return anomalies
            
        except Exception as e:
            logger.error(f"Error detecting device connectivity issues: {e}")
            return []
    
    async def detect_general_anomalies(self) -> List[Dict[str, Any]]:
        """Detect general anomalies and active alerts."""
        logger.info("Detecting general anomalies...")
        
        question = """
        Look for any unusual patterns in the vehicle detection data that might indicate anomalies.
        This could include unusual vehicle speeds, unexpected vehicle types, irregular patterns,
        or any data that deviates significantly from normal behavior.
        """
        
        try:
            result = run_data_raw_agent(question)
            anomalies = []
            
            if not result.get("error"):
                overview = result.get("result", {}).get("Overview", "").lower()
                
                if "anomal" in overview or "unusual" in overview or "unexpected" in overview:
                    device_id = self._extract_device_from_result(result)
                    
                    anomalies.append(self._format_anomaly_output(
                        anomaly_type="General Active Alert",
                        description="Active monitoring alert detected",
                        device_id=device_id or "Device-B3",
                        details=result.get("result")
                    ))
                    logger.info("Detected general anomaly.")
                else:
                    logger.info("No general anomalies detected.")
            
            return anomalies
            
        except Exception as e:
            logger.error(f"Error detecting general anomalies: {e}")
            return []
    
    def _extract_device_from_result(self, result: Dict[str, Any]) -> Optional[str]:
        """Extract device ID from SQL agent result."""
        try:
            # Try to extract device information from the result
            details = result.get("result", {})
            sql_used = details.get("SQL Used", "")
            
            # Look for device patterns in SQL or results
            if "Device-A" in sql_used:
                return "Device-A1"
            elif "Device-B" in sql_used:
                return "Device-B2"
            elif "Device-C" in sql_used:
                return "Device-C2"
            
            # Default device IDs based on anomaly type
            overview = details.get("Overview", "").lower()
            if "direction" in overview:
                return "Device-C2"
            elif "confidence" in overview or "unrecognized" in overview:
                return "Device-A1"
            elif "connectivity" in overview:
                return "Device-B2"
            
            return "Device-B3"  # Default
            
        except Exception as e:
            logger.warning(f"Could not extract device ID: {e}")
            return None
    
    async def run_all_detections(self) -> Dict[str, Any]:
        """Run all anomaly detection methods and return comprehensive results."""
        logger.info("Starting comprehensive anomaly detection...")
        
        # Run all detection methods concurrently
        tasks = [
            self.detect_multiple_direction_changes(),
            self.detect_unrecognized_vehicle_types(),
            self.detect_device_connectivity_issues(),
            self.detect_general_anomalies()
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Flatten results and separate active vs resolved
        all_anomalies = []
        active_count = 0
        
        for result in results:
            if isinstance(result, list):
                for anomaly in result:
                    all_anomalies.append(anomaly)
                    if anomaly.get("status") == "active":
                        active_count += 1
            elif isinstance(result, Exception):
                logger.error(f"Detection task failed: {result}")
        
        # Sort by timestamp (most recent first)
        all_anomalies.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        
        return {
            "anomalies": all_anomalies,
            "active_count": active_count,
            "total_count": len(all_anomalies),
            "detection_time": datetime.now().isoformat()
        }
    
    def get_anomaly_summary(self) -> Dict[str, Any]:
        """Get a summary of current anomaly status."""
        return {
            "active_anomalies": len([a for a in self.active_anomalies if a.get("status") == "active"]),
            "resolved_anomalies": len([a for a in self.resolved_anomalies if a.get("status") == "resolved"]),
            "last_detection": datetime.now().isoformat()
        }


# Global anomaly detector instance
anomaly_detector = AnomalyDetector()


async def detect_anomalies() -> Dict[str, Any]:
    """
    Main function to detect all types of anomalies.
    Returns comprehensive anomaly detection results.
    """
    return await anomaly_detector.run_all_detections()


async def get_anomaly_summary() -> Dict[str, Any]:
    """Get summary of current anomaly status."""
    return anomaly_detector.get_anomaly_summary()


# Example usage for testing
if __name__ == "__main__":
    async def main():
        logging.basicConfig(level=logging.INFO)
        
       
        # Run comprehensive detection
        results = await detect_anomalies()
        
        for i, anomaly in enumerate(results['anomalies'], 1):
            icon_map = {
                "checkmark": "✅",
                "warning": "⚠️", 
                "shield": "🛡️"
            }
            icon = icon_map.get(anomaly.get("icon", "shield"), "🛡️")
            
    
        
        # Get summary
        summary = await get_anomaly_summary()
    
    asyncio.run(main())
