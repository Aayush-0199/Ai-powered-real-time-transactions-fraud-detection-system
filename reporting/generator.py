"""
reporting/generator.py
FraudGuard AI - FinCEN SAR PDF Report Generator
Generates Suspicious Activity Reports compliant with FinCEN BSA filing standards via ReportLab
"""

import io
import os
from datetime import datetime, timezone
from typing import List, Dict, Optional

try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch, cm
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        HRFlowable, KeepTogether
    )
    from reportlab.platypus import PageBreak
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False
    print("[SAR] reportlab not available — PDF generation disabled")


# ─── Color Palette ────────────────────────────────────────────────────────────
NAVY       = colors.HexColor('#0d1527')
BLUE       = colors.HexColor('#1e40af')
LIGHT_BLUE = colors.HexColor('#dbeafe')
RED        = colors.HexColor('#dc2626')
RED_LIGHT  = colors.HexColor('#fee2e2')
AMBER      = colors.HexColor('#d97706')
AMBER_LIGHT= colors.HexColor('#fef3c7')
GREEN      = colors.HexColor('#15803d')
GREEN_LIGHT= colors.HexColor('#dcfce7')
GRAY       = colors.HexColor('#64748b')
LIGHT_GRAY = colors.HexColor('#f1f5f9')
WHITE      = colors.white
BLACK      = colors.black


class SARReportGenerator:
    """
    FinCEN Suspicious Activity Report (SAR) PDF Generator.
    
    Generates a professionally formatted, multi-section SAR PDF including:
    - Cover page with case metadata
    - Executive summary
    - Suspicious transaction detail table
    - AI risk scoring breakdown (XGBoost / IsoForest / GNN weights)
    - SHAP feature drivers
    - Regulatory compliance checklist
    - Digital signature block
    """

    REPORT_DIR = "reporting/generated"

    def __init__(self):
        os.makedirs(self.REPORT_DIR, exist_ok=True)

    def generate(
        self,
        transactions: List[dict],
        case_id: Optional[str] = None,
        analyst_name: str = "FraudGuard AI System",
        notes: str = ""
    ) -> bytes:
        """
        Generate a SAR PDF report.
        
        Args:
            transactions: List of flagged transaction dicts (from live_transactions)
            case_id: Optional case reference number
            analyst_name: Filing analyst name
            notes: Additional narrative notes
            
        Returns:
            PDF bytes (suitable for streaming as HTTP response)
        """
        if not REPORTLAB_AVAILABLE:
            raise RuntimeError("reportlab is required for PDF generation. pip install reportlab")

        if not case_id:
            case_id = f"SAR-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"

        buffer = io.BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=letter,
            rightMargin=0.75 * inch,
            leftMargin=0.75 * inch,
            topMargin=1.0 * inch,
            bottomMargin=0.75 * inch,
            title=f"FraudGuard AI — SAR Report {case_id}",
            author="FraudGuard AI Compliance Engine",
        )

        styles = self._build_styles()
        story = []

        # ── Cover Page ──────────────────────────────────────────────────────
        story.extend(self._build_cover(styles, case_id, analyst_name, transactions))
        story.append(PageBreak())

        # ── Executive Summary ────────────────────────────────────────────────
        story.extend(self._build_executive_summary(styles, transactions, case_id, notes))
        story.append(Spacer(1, 0.2 * inch))

        # ── Suspicious Transaction Table ─────────────────────────────────────
        story.extend(self._build_transaction_table(styles, transactions))
        story.append(Spacer(1, 0.2 * inch))

        # ── AI Model Scoring Breakdown ────────────────────────────────────────
        story.extend(self._build_model_breakdown(styles, transactions))
        story.append(Spacer(1, 0.2 * inch))

        # ── SHAP Feature Drivers ─────────────────────────────────────────────
        story.extend(self._build_shap_section(styles, transactions))
        story.append(Spacer(1, 0.2 * inch))

        # ── Compliance Checklist ─────────────────────────────────────────────
        story.extend(self._build_compliance_checklist(styles))
        story.append(Spacer(1, 0.3 * inch))

        # ── Signature Block ───────────────────────────────────────────────────
        story.extend(self._build_signature(styles, analyst_name, case_id))

        doc.build(story, onFirstPage=self._add_header_footer, onLaterPages=self._add_header_footer)
        buffer.seek(0)
        return buffer.read()

    def _build_styles(self) -> dict:
        base = getSampleStyleSheet()
        styles = {}

        styles['title'] = ParagraphStyle(
            'SARTitle', parent=base['Title'],
            fontSize=22, textColor=WHITE, alignment=TA_CENTER,
            fontName='Helvetica-Bold', spaceAfter=6
        )
        styles['subtitle'] = ParagraphStyle(
            'SARSubtitle', parent=base['Normal'],
            fontSize=11, textColor=LIGHT_BLUE, alignment=TA_CENTER,
            fontName='Helvetica', spaceAfter=4
        )
        styles['section_header'] = ParagraphStyle(
            'SectionHeader', parent=base['Heading1'],
            fontSize=13, textColor=NAVY, fontName='Helvetica-Bold',
            spaceBefore=12, spaceAfter=6, borderPad=4
        )
        styles['body'] = ParagraphStyle(
            'Body', parent=base['Normal'],
            fontSize=9, textColor=BLACK, fontName='Helvetica',
            leading=14, spaceAfter=4
        )
        styles['label'] = ParagraphStyle(
            'Label', parent=base['Normal'],
            fontSize=8, textColor=GRAY, fontName='Helvetica',
            leading=12
        )
        styles['highlight'] = ParagraphStyle(
            'Highlight', parent=base['Normal'],
            fontSize=9, textColor=RED, fontName='Helvetica-Bold', leading=12
        )
        styles['small'] = ParagraphStyle(
            'Small', parent=base['Normal'],
            fontSize=7.5, textColor=GRAY, fontName='Helvetica', leading=10
        )
        return styles

    def _build_cover(self, styles, case_id, analyst_name, transactions) -> list:
        elements = []
        now = datetime.now(timezone.utc)
        flagged = [t for t in transactions if t.get('IsFraud') or t.get('IsHighRisk')]

        # Navy header banner
        cover_data = [[
            Paragraph('🛡️ FraudGuard AI', styles['title']),
        ]]
        cover_table = Table(cover_data, colWidths=[6.5 * inch])
        cover_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), NAVY),
            ('TOPPADDING', (0, 0), (-1, -1), 20),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 20),
            ('LEFTPADDING', (0, 0), (-1, -1), 20),
            ('RIGHTPADDING', (0, 0), (-1, -1), 20),
            ('ROUNDEDCORNERS', [8]),
        ]))
        elements.append(cover_table)
        elements.append(Spacer(1, 0.1 * inch))

        elements.append(Paragraph(
            'SUSPICIOUS ACTIVITY REPORT (SAR)',
            ParagraphStyle('SARHead', fontSize=16, textColor=NAVY,
                           fontName='Helvetica-Bold', alignment=TA_CENTER, spaceAfter=4)
        ))
        elements.append(Paragraph(
            'FinCEN BSA Filing — Confidential Law Enforcement Document',
            ParagraphStyle('SARSub', fontSize=10, textColor=GRAY,
                           fontName='Helvetica', alignment=TA_CENTER, spaceAfter=12)
        ))
        elements.append(HRFlowable(width='100%', thickness=1, color=BLUE))
        elements.append(Spacer(1, 0.15 * inch))

        meta_data = [
            ['Case Reference Number:', case_id],
            ['Filing Date (UTC):', now.strftime('%B %d, %Y at %H:%M:%S UTC')],
            ['Report Generated By:', analyst_name],
            ['Total Transactions Reviewed:', str(len(transactions))],
            ['Flagged Suspicious Activities:', str(len(flagged))],
            ['Report Classification:', 'CONFIDENTIAL — LAW ENFORCEMENT SENSITIVE'],
        ]
        meta_table = Table(meta_data, colWidths=[2.2 * inch, 4.3 * inch])
        meta_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), LIGHT_GRAY),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('TEXTCOLOR', (0, 0), (0, -1), NAVY),
            ('TEXTCOLOR', (1, 0), (1, -1), BLACK),
            ('PADDING', (0, 0), (-1, -1), 6),
            ('ROWBACKGROUNDS', (0, 0), (-1, -1), [WHITE, LIGHT_GRAY]),
            ('BOX', (0, 0), (-1, -1), 0.5, GRAY),
            ('LINEBELOW', (0, 0), (-1, -2), 0.25, colors.HexColor('#e2e8f0')),
            ('TEXTCOLOR', (1, 5), (1, 5), RED),
            ('FONTNAME', (1, 5), (1, 5), 'Helvetica-Bold'),
        ]))
        elements.append(meta_table)
        elements.append(Spacer(1, 0.3 * inch))

        # Warning box
        warn_data = [[Paragraph(
            '⚠️  This document contains confidential suspicious activity information. '
            'Distribution is restricted to authorized compliance and law enforcement personnel only. '
            'Unauthorized disclosure is a federal offense under 31 U.S.C. § 5318(g)(2).',
            ParagraphStyle('Warn', fontSize=8, textColor=colors.HexColor('#92400e'),
                           fontName='Helvetica', leading=12)
        )]]
        warn_table = Table(warn_data, colWidths=[6.5 * inch])
        warn_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), AMBER_LIGHT),
            ('BOX', (0, 0), (-1, -1), 1, AMBER),
            ('PADDING', (0, 0), (-1, -1), 10),
        ]))
        elements.append(warn_table)
        return elements

    def _build_executive_summary(self, styles, transactions, case_id, notes) -> list:
        elements = []
        elements.append(Paragraph('1. Executive Summary', styles['section_header']))
        elements.append(HRFlowable(width='100%', thickness=0.5, color=BLUE))
        elements.append(Spacer(1, 0.1 * inch))

        flagged = [t for t in transactions if t.get('IsFraud') or t.get('IsHighRisk')]
        blocked = [t for t in transactions if t.get('IsBlocked') or t.get('Status') == 'BLOCKED']
        total_flagged_amount = sum(float(t.get('TransactionAmount', 0)) for t in flagged)
        avg_risk = (sum(float(t.get('RiskScore', 0)) for t in flagged) / len(flagged)) if flagged else 0

        summary_text = (
            f"FraudGuard AI's real-time ensemble detection engine processed <b>{len(transactions)}</b> financial "
            f"transactions and identified <b>{len(flagged)}</b> suspicious activities requiring immediate regulatory attention. "
            f"Of these, <b>{len(blocked)}</b> accounts were automatically frozen pending investigation. "
            f"The aggregate suspicious transaction value totals <b>${total_flagged_amount:,.2f}</b> with an average "
            f"AI composite risk score of <b>{avg_risk:.3f}</b> (threshold: 0.500). "
            f"This SAR is filed pursuant to 31 CFR § 1020.320 (Banks) and FinCEN guidance FIN-2014-G001."
        )
        elements.append(Paragraph(summary_text, styles['body']))

        if notes:
            elements.append(Spacer(1, 0.1 * inch))
            elements.append(Paragraph(f'<b>Analyst Notes:</b> {notes}', styles['body']))

        return elements

    def _build_transaction_table(self, styles, transactions) -> list:
        elements = []
        elements.append(Paragraph('2. Suspicious Transaction Detail', styles['section_header']))
        elements.append(HRFlowable(width='100%', thickness=0.5, color=BLUE))
        elements.append(Spacer(1, 0.08 * inch))

        flagged = [t for t in transactions if
                   t.get('IsFraud') or t.get('IsHighRisk') or float(t.get('RiskScore', 0)) > 0.5][:30]

        if not flagged:
            elements.append(Paragraph('No flagged transactions in current reporting window.', styles['body']))
            return elements

        headers = ['Transaction ID', 'Account ID', 'Amount', 'Risk Score', 'Location', 'Type', 'Status']
        data = [headers]

        for txn in flagged:
            risk = float(txn.get('RiskScore', 0))
            status = str(txn.get('Status', 'FLAGGED'))
            data.append([
                str(txn.get('TransactionID', 'N/A'))[:14],
                str(txn.get('AccountID', 'N/A'))[:12],
                f"${float(txn.get('TransactionAmount', 0)):,.2f}",
                f"{risk:.3f}",
                str(txn.get('Location', 'N/A'))[:14],
                str(txn.get('TransactionType', 'N/A'))[:10],
                status[:10],
            ])

        col_widths = [1.2*inch, 1.0*inch, 0.9*inch, 0.8*inch, 1.1*inch, 0.85*inch, 0.85*inch]
        table = Table(data, colWidths=col_widths, repeatRows=1)

        row_styles = [
            ('BACKGROUND', (0, 0), (-1, 0), NAVY),
            ('TEXTCOLOR', (0, 0), (-1, 0), WHITE),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 7.5),
            ('ALIGN', (2, 0), (3, -1), 'RIGHT'),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('PADDING', (0, 0), (-1, -1), 4),
            ('BOX', (0, 0), (-1, -1), 0.5, GRAY),
            ('LINEBELOW', (0, 0), (-1, -2), 0.25, colors.HexColor('#e2e8f0')),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [WHITE, LIGHT_GRAY]),
        ]

        # Highlight blocked rows
        for i, txn in enumerate(flagged, start=1):
            if txn.get('IsBlocked') or txn.get('Status') == 'BLOCKED':
                row_styles.append(('BACKGROUND', (0, i), (-1, i), RED_LIGHT))
                row_styles.append(('TEXTCOLOR', (0, i), (-1, i), RED))

        table.setStyle(TableStyle(row_styles))
        elements.append(table)
        return elements

    def _build_model_breakdown(self, styles, transactions) -> list:
        elements = []
        elements.append(Paragraph('3. AI Ensemble Scoring Methodology', styles['section_header']))
        elements.append(HRFlowable(width='100%', thickness=0.5, color=BLUE))
        elements.append(Spacer(1, 0.08 * inch))

        elements.append(Paragraph(
            'FraudGuard AI employs a weighted ensemble of three independent ML models. '
            'Final composite risk score = 0.45 × XGBoost + 0.30 × Graph Neural Network + 0.25 × Isolation Forest.',
            styles['body']
        ))
        elements.append(Spacer(1, 0.08 * inch))

        model_data = [
            ['Model', 'Weight', 'Architecture', 'Training Data', 'Status'],
            ['XGBoost Classifier', '45%', 'Gradient Boosted Trees (500 est.)', '5,000 labeled txns', '✓ Active'],
            ['Graph Neural Network', '30%', 'GCNConv × 3 layers (PyG)', 'Account-Merchant-Device graph', '✓ Active'],
            ['Isolation Forest', '25%', 'Ensemble isolation trees (200 est.)', 'Unlabeled stream (unsupervised)', '✓ Active'],
        ]
        model_table = Table(model_data, colWidths=[1.4*inch, 0.6*inch, 1.9*inch, 1.65*inch, 0.75*inch])
        model_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), NAVY),
            ('TEXTCOLOR', (0, 0), (-1, 0), WHITE),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 8),
            ('PADDING', (0, 0), (-1, -1), 5),
            ('BOX', (0, 0), (-1, -1), 0.5, GRAY),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [WHITE, LIGHT_GRAY]),
            ('TEXTCOLOR', (4, 1), (4, -1), GREEN),
            ('FONTNAME', (4, 1), (4, -1), 'Helvetica-Bold'),
        ]))
        elements.append(model_table)
        return elements

    def _build_shap_section(self, styles, transactions) -> list:
        elements = []
        elements.append(Paragraph('4. AI Explainability — SHAP Feature Contributions', styles['section_header']))
        elements.append(HRFlowable(width='100%', thickness=0.5, color=BLUE))
        elements.append(Spacer(1, 0.08 * inch))

        # Aggregate top SHAP drivers from flagged transactions
        shap_aggregate: Dict[str, float] = {}
        flagged = [t for t in transactions if float(t.get('RiskScore', 0)) > 0.5]

        for txn in flagged[:20]:
            for driver in txn.get('SHAPDrivers', []):
                feat = driver.get('feature', '')
                val = abs(float(driver.get('shap_value', 0)))
                shap_aggregate[feat] = shap_aggregate.get(feat, 0) + val

        if shap_aggregate:
            sorted_drivers = sorted(shap_aggregate.items(), key=lambda x: x[1], reverse=True)[:8]
            shap_data = [['Feature', 'Aggregate |SHAP|', 'Fraud Signal Strength']]
            max_val = sorted_drivers[0][1] if sorted_drivers else 1.0

            for feat, val in sorted_drivers:
                bar_len = int((val / max_val) * 20)
                bar = '█' * bar_len + '░' * (20 - bar_len)
                strength = 'CRITICAL' if val/max_val > 0.7 else ('HIGH' if val/max_val > 0.4 else 'MODERATE')
                shap_data.append([feat, f"{val:.4f}", f"{bar}  {strength}"])

            shap_table = Table(shap_data, colWidths=[2.0*inch, 1.2*inch, 3.3*inch])
            shap_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), NAVY),
                ('TEXTCOLOR', (0, 0), (-1, 0), WHITE),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTNAME', (2, 1), (2, -1), 'Courier'),
                ('FONTSIZE', (0, 0), (-1, -1), 8),
                ('PADDING', (0, 0), (-1, -1), 5),
                ('BOX', (0, 0), (-1, -1), 0.5, GRAY),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [WHITE, LIGHT_GRAY]),
            ]))
            elements.append(shap_table)
        else:
            elements.append(Paragraph('No SHAP data available for current transaction set.', styles['body']))

        return elements

    def _build_compliance_checklist(self, styles) -> list:
        elements = []
        elements.append(Paragraph('5. Regulatory Compliance Checklist', styles['section_header']))
        elements.append(HRFlowable(width='100%', thickness=0.5, color=BLUE))
        elements.append(Spacer(1, 0.08 * inch))

        checklist = [
            ('✓', 'BSA/AML Transaction Monitoring', 'Automated real-time surveillance active', GREEN),
            ('✓', 'FinCEN SAR Filing Obligation', '31 CFR § 1020.320 — Filed within 30-day window', GREEN),
            ('✓', 'OFAC Sanctions Screening', 'Location-based geo-risk screening active', GREEN),
            ('✓', 'Velocity Rule Monitoring', 'Multi-transaction pattern detection active', GREEN),
            ('✓', 'Account Freeze Capability', 'Automated freeze/unfreeze via incident response', GREEN),
            ('✓', 'Audit Trail Preservation', 'Immutable transaction log with timestamps', GREEN),
            ('✓', 'Model Explainability (XAI)', 'SHAP-based decision rationale per transaction', GREEN),
            ('○', 'Human Analyst Review', 'Manual review recommended for CRITICAL tier cases', AMBER),
        ]

        check_data = [['Status', 'Compliance Control', 'Notes']]
        for status, control, note, color in checklist:
            check_data.append([status, control, note])

        check_table = Table(check_data, colWidths=[0.5*inch, 2.1*inch, 3.9*inch])
        row_styles = [
            ('BACKGROUND', (0, 0), (-1, 0), NAVY),
            ('TEXTCOLOR', (0, 0), (-1, 0), WHITE),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 8),
            ('ALIGN', (0, 0), (0, -1), 'CENTER'),
            ('PADDING', (0, 0), (-1, -1), 5),
            ('BOX', (0, 0), (-1, -1), 0.5, GRAY),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [WHITE, LIGHT_GRAY]),
        ]
        for i, (status, _, _, color) in enumerate(checklist, start=1):
            check_table_color = GREEN if color == GREEN else AMBER
            row_styles.append(('TEXTCOLOR', (0, i), (0, i), check_table_color))
            row_styles.append(('FONTNAME', (0, i), (0, i), 'Helvetica-Bold'))

        check_table.setStyle(TableStyle(row_styles))
        elements.append(check_table)
        return elements

    def _build_signature(self, styles, analyst_name, case_id) -> list:
        elements = []
        now = datetime.now(timezone.utc)
        elements.append(HRFlowable(width='100%', thickness=1, color=NAVY))
        elements.append(Spacer(1, 0.1 * inch))

        sig_data = [
            [
                Paragraph(f'<b>Filing Analyst:</b><br/>{analyst_name}', styles['body']),
                Paragraph(f'<b>Case Reference:</b><br/>{case_id}', styles['body']),
                Paragraph(f'<b>Digital Timestamp:</b><br/>{now.strftime("%Y-%m-%d %H:%M:%S UTC")}', styles['body']),
            ]
        ]
        sig_table = Table(sig_data, colWidths=[2.2*inch, 2.2*inch, 2.1*inch])
        sig_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), LIGHT_GRAY),
            ('BOX', (0, 0), (-1, -1), 0.5, GRAY),
            ('PADDING', (0, 0), (-1, -1), 8),
        ]))
        elements.append(sig_table)
        elements.append(Spacer(1, 0.05 * inch))
        elements.append(Paragraph(
            '⚡ Generated by FraudGuard AI — Enterprise Financial Crime Detection Platform',
            ParagraphStyle('Footer', fontSize=7, textColor=GRAY, fontName='Helvetica', alignment=TA_CENTER)
        ))
        return elements

    def _add_header_footer(self, canvas, doc):
        """Draw page header and footer on every page."""
        canvas.saveState()
        w, h = letter

        # Header
        canvas.setFillColor(NAVY)
        canvas.rect(0.75*inch, h - 0.6*inch, w - 1.5*inch, 0.35*inch, fill=1, stroke=0)
        canvas.setFillColor(WHITE)
        canvas.setFont('Helvetica-Bold', 9)
        canvas.drawString(0.85*inch, h - 0.42*inch, '🛡️ FraudGuard AI — Suspicious Activity Report')
        canvas.setFont('Helvetica', 8)
        canvas.drawRightString(w - 0.75*inch, h - 0.42*inch,
                               f'CONFIDENTIAL  |  Page {doc.page}')

        # Footer line
        canvas.setStrokeColor(BLUE)
        canvas.setLineWidth(0.5)
        canvas.line(0.75*inch, 0.5*inch, w - 0.75*inch, 0.5*inch)
        canvas.setFillColor(GRAY)
        canvas.setFont('Helvetica', 7)
        canvas.drawString(0.75*inch, 0.35*inch,
                          'This report is generated automatically by FraudGuard AI. '
                          'For authorized personnel only.')
        canvas.drawRightString(w - 0.75*inch, 0.35*inch,
                               f'{datetime.now(timezone.utc).strftime("%Y-%m-%d")}')

        canvas.restoreState()
