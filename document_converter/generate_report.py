"""
Generate executive summary report for document processing
"""
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging

from .quality_checker import QualityChecker

logger = logging.getLogger(__name__)


class ReportGenerator:
    """Generate executive summary reports"""
    
    def __init__(self, log_file: Optional[Path] = None):
        """
        Initialize report generator
        
        Args:
            log_file: Path to processing log file
        """
        self.log_file = log_file or Path("Output/ProcessingLog.json")
        self.quality_checker = QualityChecker()
    
    def generate_report(self, output_path: Optional[Path] = None) -> Dict[str, Any]:
        """
        Generate executive summary report
        
        Args:
            output_path: Optional path to save report
            
        Returns:
            Report dictionary
        """
        # Load processing log
        processing_data = self._load_processing_log()
        
        # Calculate statistics
        stats = self._calculate_statistics(processing_data)
        
        # Generate report
        report = {
            "report_date": datetime.now().isoformat(),
            "summary": {
                "total_files_processed": stats["total_files"],
                "successful_extractions": stats["successful"],
                "failed_extractions": stats["failed"],
                "skipped_files": stats["skipped"],
                "success_rate": stats["success_rate"],
                "average_items_per_file": stats["avg_items"],
                "total_items_extracted": stats["total_items"]
            },
            "classification_metrics": {
                "stock_sales_reports": stats["stock_count"],
                "other_documents": stats["other_count"],
                "classification_accuracy": stats.get("classification_accuracy", 0.0)
            },
            "extraction_quality": {
                "files_with_zero_items": stats["zero_items_count"],
                "zero_items_rate": stats["zero_items_rate"],
                "average_quality_score": stats.get("avg_quality_score", 0.0)
            },
            "performance_metrics": {
                "average_processing_time": stats.get("avg_processing_time", 0.0),
                "total_processing_time": stats.get("total_processing_time", 0.0)
            },
            "common_issues": stats.get("common_issues", []),
            "recommendations": self._generate_recommendations(stats),
            "detailed_statistics": stats
        }
        
        # Save report if output path provided
        if output_path:
            self._save_report(report, output_path)
        
        return report
    
    def _load_processing_log(self) -> List[Dict[str, Any]]:
        """Load processing log data"""
        try:
            if self.log_file.exists():
                with open(self.log_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        return data
                    elif isinstance(data, dict) and "entries" in data:
                        return data["entries"]
                    else:
                        return []
            else:
                logger.warning(f"Processing log not found: {self.log_file}")
                return []
        except Exception as e:
            logger.error(f"Error loading processing log: {e}")
            return []
    
    def _calculate_statistics(self, processing_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate statistics from processing data"""
        stats = {
            "total_files": len(processing_data),
            "successful": 0,
            "failed": 0,
            "skipped": 0,
            "stock_count": 0,
            "other_count": 0,
            "total_items": 0,
            "zero_items_count": 0,
            "files_with_items": 0,
            "processing_times": [],
            "quality_scores": [],
            "common_issues": []
        }
        
        issues_count = {}
        
        for entry in processing_data:
            status = entry.get("status", "unknown")
            classification = entry.get("classification", "unknown")
            items_count = entry.get("items_count", 0) or 0
            
            # Count by status
            if status == "success":
                stats["successful"] += 1
            elif status == "error":
                stats["failed"] += 1
            elif status == "skipped":
                stats["skipped"] += 1
            
            # Count by classification
            if classification == "stock_sales_report":
                stats["stock_count"] += 1
            elif classification == "other":
                stats["other_count"] += 1
            
            # Count items
            if items_count > 0:
                stats["total_items"] += items_count
                stats["files_with_items"] += 1
            else:
                if classification == "stock_sales_report":
                    stats["zero_items_count"] += 1
            
            # Track processing time if available
            processing_time = entry.get("processing_time")
            if processing_time:
                stats["processing_times"].append(processing_time)
            
            # Track quality score if available
            quality_score = entry.get("quality_score")
            if quality_score:
                stats["quality_scores"].append(quality_score)
            
            # Track common issues
            error = entry.get("error", "")
            if error:
                error_key = error[:50]  # First 50 chars as key
                issues_count[error_key] = issues_count.get(error_key, 0) + 1
        
        # Calculate rates
        if stats["total_files"] > 0:
            stats["success_rate"] = stats["successful"] / stats["total_files"]
            stats["zero_items_rate"] = stats["zero_items_count"] / stats["stock_count"] if stats["stock_count"] > 0 else 0
        else:
            stats["success_rate"] = 0.0
            stats["zero_items_rate"] = 0.0
        
        # Calculate averages
        if stats["files_with_items"] > 0:
            stats["avg_items"] = stats["total_items"] / stats["files_with_items"]
        else:
            stats["avg_items"] = 0.0
        
        if stats["processing_times"]:
            stats["avg_processing_time"] = sum(stats["processing_times"]) / len(stats["processing_times"])
            stats["total_processing_time"] = sum(stats["processing_times"])
        else:
            stats["avg_processing_time"] = 0.0
            stats["total_processing_time"] = 0.0
        
        if stats["quality_scores"]:
            stats["avg_quality_score"] = sum(stats["quality_scores"]) / len(stats["quality_scores"])
        else:
            stats["avg_quality_score"] = 0.0
        
        # Get top common issues
        stats["common_issues"] = sorted(
            issues_count.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]  # Top 10 issues
        
        return stats
    
    def _generate_recommendations(self, stats: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on statistics"""
        recommendations = []
        
        # Success rate recommendations
        if stats["success_rate"] < 0.8:
            recommendations.append(
                f"Success rate is {stats['success_rate']:.1%} - consider reviewing failed files and improving extraction strategies"
            )
        
        # Zero items recommendations
        if stats["zero_items_rate"] > 0.1:
            recommendations.append(
                f"{stats['zero_items_rate']:.1%} of Stock files extracted 0 items - review extraction logic for these files"
            )
        
        # Quality score recommendations
        if stats.get("avg_quality_score", 1.0) < 0.7:
            recommendations.append(
                f"Average quality score is {stats['avg_quality_score']:.2f} - data quality may need improvement"
            )
        
        # Performance recommendations
        if stats.get("avg_processing_time", 0) > 10:
            recommendations.append(
                f"Average processing time is {stats['avg_processing_time']:.1f}s - consider optimization"
            )
        
        # Common issues recommendations
        if stats["common_issues"]:
            top_issue = stats["common_issues"][0]
            if top_issue[1] > stats["total_files"] * 0.1:  # Affects >10% of files
                recommendations.append(
                    f"Common issue: '{top_issue[0]}' affects {top_issue[1]} files - investigate root cause"
                )
        
        if not recommendations:
            recommendations.append("System is performing well - no major issues detected")
        
        return recommendations
    
    def _save_report(self, report: Dict[str, Any], output_path: Path):
        """Save report to file"""
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save as JSON
            json_path = output_path.with_suffix('.json')
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            
            # Save as text summary
            txt_path = output_path.with_suffix('.txt')
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(self._format_text_report(report))
            
            logger.info(f"Report saved to {json_path} and {txt_path}")
        except Exception as e:
            logger.error(f"Error saving report: {e}")
    
    def _format_text_report(self, report: Dict[str, Any]) -> str:
        """Format report as text"""
        lines = []
        lines.append("=" * 80)
        lines.append("DOCUMENT PROCESSING EXECUTIVE SUMMARY REPORT")
        lines.append("=" * 80)
        lines.append(f"Report Date: {report['report_date']}")
        lines.append("")
        
        # Summary
        summary = report["summary"]
        lines.append("SUMMARY")
        lines.append("-" * 80)
        lines.append(f"Total Files Processed: {summary['total_files_processed']}")
        lines.append(f"Successful Extractions: {summary['successful_extractions']}")
        lines.append(f"Failed Extractions: {summary['failed_extractions']}")
        lines.append(f"Skipped Files: {summary['skipped_files']}")
        lines.append(f"Success Rate: {summary['success_rate']:.1%}")
        lines.append(f"Total Items Extracted: {summary['total_items_extracted']}")
        lines.append(f"Average Items per File: {summary['average_items_per_file']:.1f}")
        lines.append("")
        
        # Classification
        class_metrics = report["classification_metrics"]
        lines.append("CLASSIFICATION METRICS")
        lines.append("-" * 80)
        lines.append(f"Stock & Sales Reports: {class_metrics['stock_sales_reports']}")
        lines.append(f"Other Documents: {class_metrics['other_documents']}")
        lines.append("")
        
        # Quality
        quality = report["extraction_quality"]
        lines.append("EXTRACTION QUALITY")
        lines.append("-" * 80)
        lines.append(f"Files with Zero Items: {quality['files_with_zero_items']}")
        lines.append(f"Zero Items Rate: {quality['zero_items_rate']:.1%}")
        lines.append(f"Average Quality Score: {quality['average_quality_score']:.2f}")
        lines.append("")
        
        # Performance
        perf = report["performance_metrics"]
        lines.append("PERFORMANCE METRICS")
        lines.append("-" * 80)
        lines.append(f"Average Processing Time: {perf['average_processing_time']:.2f}s")
        lines.append(f"Total Processing Time: {perf['total_processing_time']:.2f}s")
        lines.append("")
        
        # Recommendations
        lines.append("RECOMMENDATIONS")
        lines.append("-" * 80)
        for i, rec in enumerate(report["recommendations"], 1):
            lines.append(f"{i}. {rec}")
        lines.append("")
        
        # Common Issues
        if report["common_issues"]:
            lines.append("COMMON ISSUES")
            lines.append("-" * 80)
            for issue, count in report["common_issues"][:5]:
                lines.append(f"- {issue} ({count} occurrences)")
            lines.append("")
        
        lines.append("=" * 80)
        return "\n".join(lines)


def generate_report(log_file: Optional[Path] = None, output_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Convenience function to generate report
    
    Args:
        log_file: Path to processing log file
        output_path: Path to save report
        
    Returns:
        Report dictionary
    """
    generator = ReportGenerator(log_file)
    return generator.generate_report(output_path)

