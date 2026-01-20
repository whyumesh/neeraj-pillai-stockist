"""
Quality checker for extracted data
Validates completeness, consistency, and flags suspicious extractions
"""
from typing import Dict, List, Any, Optional
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class QualityChecker:
    """Quality checker for extracted stock sales data"""
    
    def __init__(self):
        """Initialize quality checker"""
        self.quality_scores = {}
    
    def check_extraction_quality(self, extracted_data: Dict[str, Any], 
                                pdf_path: Optional[Path] = None) -> Dict[str, Any]:
        """
        Check quality of extracted data
        
        Args:
            extracted_data: Extracted data dictionary
            pdf_path: Optional path to source PDF for context
            
        Returns:
            Quality report dictionary
        """
        report = {
            "overall_score": 0.0,
            "checks": {},
            "warnings": [],
            "errors": [],
            "suggestions": []
        }
        
        items = extracted_data.get("items", [])
        sections = extracted_data.get("sections", [])
        period = extracted_data.get("period")
        diagnostics = extracted_data.get("diagnostics", {})
        
        # Check 1: Item count
        item_count_score = self._check_item_count(items)
        report["checks"]["item_count"] = item_count_score
        
        # Check 2: Data completeness
        completeness_score = self._check_completeness(items)
        report["checks"]["completeness"] = completeness_score
        
        # Check 3: Data consistency
        consistency_score = self._check_consistency(items)
        report["checks"]["consistency"] = consistency_score
        
        # Check 4: Field validation
        field_validation_score = self._check_field_validation(items)
        report["checks"]["field_validation"] = field_validation_score
        
        # Check 5: Section detection
        section_score = self._check_sections(sections)
        report["checks"]["sections"] = section_score
        
        # Check 6: Period detection
        period_score = self._check_period(period)
        report["checks"]["period"] = period_score
        
        # Check 7: Extraction diagnostics
        diagnostics_score = self._check_diagnostics(diagnostics)
        report["checks"]["diagnostics"] = diagnostics_score
        
        # Calculate overall score (weighted average)
        weights = {
            "item_count": 0.25,
            "completeness": 0.25,
            "consistency": 0.15,
            "field_validation": 0.15,
            "sections": 0.05,
            "period": 0.05,
            "diagnostics": 0.10
        }
        
        overall_score = sum(
            report["checks"][key]["score"] * weights.get(key, 0)
            for key in report["checks"]
        )
        
        report["overall_score"] = overall_score
        
        # Generate warnings and suggestions
        self._generate_warnings(report, items, sections, period, diagnostics)
        
        return report
    
    def _check_item_count(self, items: List[Dict]) -> Dict[str, Any]:
        """Check if reasonable number of items extracted"""
        item_count = len(items)
        
        if item_count == 0:
            return {
                "score": 0.0,
                "status": "error",
                "message": "No items extracted",
                "item_count": 0
            }
        elif item_count < 5:
            return {
                "score": 0.3,
                "status": "warning",
                "message": f"Very few items extracted ({item_count})",
                "item_count": item_count
            }
        elif item_count < 20:
            return {
                "score": 0.7,
                "status": "warning",
                "message": f"Few items extracted ({item_count})",
                "item_count": item_count
            }
        else:
            return {
                "score": 1.0,
                "status": "ok",
                "message": f"Reasonable number of items ({item_count})",
                "item_count": item_count
            }
    
    def _check_completeness(self, items: List[Dict]) -> Dict[str, Any]:
        """Check data completeness - required fields present"""
        if not items:
            return {
                "score": 0.0,
                "status": "error",
                "message": "No items to check",
                "completeness_rate": 0.0
            }
        
        required_fields = ["Item Description", "item_description"]
        optional_fields = [
            "Opening Qty", "Opening Value",
            "Receipt Qty", "Receipt Value",
            "Issue Qty", "Issue Value",
            "Closing Qty", "Closing Value",
            "Dump Qty"
        ]
        
        items_with_desc = 0
        items_with_data = 0
        
        for item in items:
            # Check for description
            has_desc = any(item.get(field) for field in required_fields)
            if has_desc:
                items_with_desc += 1
            
            # Check for at least some numeric data
            has_data = any(
                item.get(field) not in [None, "", 0, 0.0, "-"] 
                for field in optional_fields
            )
            if has_data:
                items_with_data += 1
        
        desc_rate = items_with_desc / len(items) if items else 0
        data_rate = items_with_data / len(items) if items else 0
        
        completeness_rate = (desc_rate * 0.5) + (data_rate * 0.5)
        
        if completeness_rate >= 0.9:
            status = "ok"
            score = 1.0
        elif completeness_rate >= 0.7:
            status = "warning"
            score = 0.7
        elif completeness_rate >= 0.5:
            status = "warning"
            score = 0.5
        else:
            status = "error"
            score = 0.3
        
        return {
            "score": score,
            "status": status,
            "message": f"Completeness: {completeness_rate:.1%} (descriptions: {desc_rate:.1%}, data: {data_rate:.1%})",
            "completeness_rate": completeness_rate,
            "items_with_description": items_with_desc,
            "items_with_data": items_with_data
        }
    
    def _check_consistency(self, items: List[Dict]) -> Dict[str, Any]:
        """Check data consistency - values make sense"""
        if not items:
            return {
                "score": 1.0,
                "status": "ok",
                "message": "No items to check",
                "consistency_issues": []
            }
        
        issues = []
        
        for i, item in enumerate(items):
            # Check for negative quantities (might be valid but unusual)
            qty_fields = ["Opening Qty", "Receipt Qty", "Issue Qty", "Closing Qty", "Dump Qty"]
            for field in qty_fields:
                value = item.get(field)
                if isinstance(value, (int, float)) and value < 0:
                    issues.append(f"Item {i+1}: Negative {field}: {value}")
            
            # Check for very large values (might be errors)
            for field in qty_fields:
                value = item.get(field)
                if isinstance(value, (int, float)) and value > 1000000:
                    issues.append(f"Item {i+1}: Very large {field}: {value}")
            
            # Check for missing values in critical fields
            desc = item.get("Item Description") or item.get("item_description")
            if not desc or desc.strip() == "":
                issues.append(f"Item {i+1}: Missing description")
        
        issue_count = len(issues)
        total_checks = len(items) * 5  # Rough estimate
        
        consistency_rate = 1.0 - (issue_count / total_checks) if total_checks > 0 else 1.0
        consistency_rate = max(0.0, consistency_rate)
        
        if consistency_rate >= 0.95:
            status = "ok"
            score = 1.0
        elif consistency_rate >= 0.85:
            status = "warning"
            score = 0.8
        else:
            status = "warning"
            score = 0.6
        
        return {
            "score": score,
            "status": status,
            "message": f"Consistency: {consistency_rate:.1%} ({issue_count} issues found)",
            "consistency_rate": consistency_rate,
            "consistency_issues": issues[:10]  # Limit to first 10 issues
        }
    
    def _check_field_validation(self, items: List[Dict]) -> Dict[str, Any]:
        """Validate field formats and types"""
        if not items:
            return {
                "score": 1.0,
                "status": "ok",
                "message": "No items to validate",
                "validation_errors": []
            }
        
        errors = []
        numeric_fields = [
            "Opening Qty", "Opening Value",
            "Receipt Qty", "Receipt Value",
            "Issue Qty", "Issue Value",
            "Closing Qty", "Closing Value",
            "Dump Qty"
        ]
        
        for i, item in enumerate(items):
            for field in numeric_fields:
                value = item.get(field)
                if value is not None and value != "" and value != "-":
                    if not isinstance(value, (int, float)):
                        try:
                            float(str(value).replace(",", ""))
                        except ValueError:
                            errors.append(f"Item {i+1}, {field}: Invalid numeric value '{value}'")
        
        error_rate = len(errors) / (len(items) * len(numeric_fields)) if items else 0
        
        if error_rate == 0:
            status = "ok"
            score = 1.0
        elif error_rate < 0.05:
            status = "warning"
            score = 0.8
        else:
            status = "error"
            score = 0.5
        
        return {
            "score": score,
            "status": status,
            "message": f"Field validation: {len(errors)} errors found",
            "validation_errors": errors[:10]  # Limit to first 10 errors
        }
    
    def _check_sections(self, sections: List[str]) -> Dict[str, Any]:
        """Check if sections were detected"""
        section_count = len(sections) if sections else 0
        
        if section_count == 0:
            return {
                "score": 0.5,
                "status": "warning",
                "message": "No sections detected",
                "section_count": 0
            }
        elif section_count < 3:
            return {
                "score": 0.7,
                "status": "ok",
                "message": f"Few sections detected ({section_count})",
                "section_count": section_count
            }
        else:
            return {
                "score": 1.0,
                "status": "ok",
                "message": f"Sections detected ({section_count})",
                "section_count": section_count
            }
    
    def _check_period(self, period: Optional[str]) -> Dict[str, Any]:
        """Check if period was detected"""
        if period:
            return {
                "score": 1.0,
                "status": "ok",
                "message": f"Period detected: {period}",
                "period": period
            }
        else:
            return {
                "score": 0.5,
                "status": "warning",
                "message": "No period detected",
                "period": None
            }
    
    def _check_diagnostics(self, diagnostics: Dict[str, Any]) -> Dict[str, Any]:
        """Check extraction diagnostics"""
        if not diagnostics:
            return {
                "score": 0.5,
                "status": "warning",
                "message": "No diagnostics available",
                "diagnostics": {}
            }
        
        strategies_tried = len(diagnostics.get("strategies_tried", []))
        strategies_succeeded = len(diagnostics.get("strategies_succeeded", []))
        
        if strategies_succeeded == 0:
            return {
                "score": 0.0,
                "status": "error",
                "message": "No extraction strategies succeeded",
                "strategies_tried": strategies_tried,
                "strategies_succeeded": strategies_succeeded
            }
        
        success_rate = strategies_succeeded / strategies_tried if strategies_tried > 0 else 0
        
        if success_rate >= 0.5:
            score = 1.0
            status = "ok"
        elif success_rate > 0:
            score = 0.7
            status = "warning"
        else:
            score = 0.3
            status = "error"
        
        return {
            "score": score,
            "status": status,
            "message": f"Extraction diagnostics: {strategies_succeeded}/{strategies_tried} strategies succeeded",
            "strategies_tried": strategies_tried,
            "strategies_succeeded": strategies_succeeded,
            "success_rate": success_rate
        }
    
    def _generate_warnings(self, report: Dict[str, Any], items: List[Dict],
                          sections: List[str], period: Optional[str],
                          diagnostics: Dict[str, Any]):
        """Generate warnings and suggestions based on quality checks"""
        # Check for low overall score
        if report["overall_score"] < 0.5:
            report["warnings"].append("Low overall quality score - extraction may be incomplete")
            report["suggestions"].append("Review the source PDF manually and verify extraction")
        
        # Check for zero items
        if len(items) == 0:
            report["errors"].append("No items extracted - file may require manual processing")
            report["suggestions"].append("Try OCR if PDF is scanned, or check if file format is supported")
        
        # Check for low completeness
        completeness = report["checks"].get("completeness", {})
        if completeness.get("completeness_rate", 1.0) < 0.7:
            report["warnings"].append("Low data completeness - many items missing required fields")
            report["suggestions"].append("Check if PDF structure matches expected format")
        
        # Check for extraction issues
        diag_check = report["checks"].get("diagnostics", {})
        if diag_check.get("success_rate", 1.0) < 0.5:
            report["warnings"].append("Multiple extraction strategies failed")
            report["suggestions"].append("File may have unusual format - consider manual review")
        
        # Check for missing period
        if not period:
            report["warnings"].append("Period not detected - metadata may be incomplete")
        
        # Check for missing sections
        if len(sections) == 0:
            report["warnings"].append("No sections detected - file structure may be different")

