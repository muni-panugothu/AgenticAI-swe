/** Shared palette for dark auth screens (reference: neon lime Soft Life). */
export const AuthTheme = {
  background: '#000000',
  inputBg: '#1A1A1A',
  accent: '#D4FF00',
  textPrimary: '#FFFFFF',
  textSecondary: '#C8C8C8',
  textMuted: '#8E8E8E',
  divider: '#3F3F3F',
  error: '#FF6B6B',
  glow: 'rgba(110, 140, 20, 0.38)',
  glowSoft: 'rgba(198, 255, 0, 0.08)',
  socialBorder: '#D4FF00',
  checkboxBorder: '#FFFFFF',
  particle: 'rgba(212, 255, 0, 0.22)',
} as const;

export const LEGAL_CONTENT = {
  termsTitle: 'Terms of Use',
  privacyTitle: 'Privacy Policy',
  terms: `Last updated: November 1, 2026

1. Acceptance of Terms
By creating an account or using Soft Life planning features ("the Service"), you agree to these Terms of Use. If you do not agree, do not use the Service.

2. Eligibility
You must be at least 18 years old (or the age of majority in your jurisdiction) and able to form a binding contract to use the Service.

3. Account Security
You are responsible for safeguarding your credentials and for all activity under your account. Notify us promptly of any unauthorized use.

4. Acceptable Use
You agree not to misuse the Service, including attempting unauthorized access, disrupting systems, uploading unlawful content, or reverse engineering protected components beyond what applicable law permits.

5. AI-Assisted Features
The Service may provide AI-generated suggestions. Outputs can be incomplete or incorrect. You remain responsible for verifying advice before relying on it for financial or personal decisions.

6. Intellectual Property
The Service, branding, and software remain our property or that of our licensors. You retain rights to content you submit, and grant us a limited license to process it to operate the Service.

7. Termination
We may suspend or terminate accounts that violate these Terms or present security risk. You may stop using the Service at any time.

8. Disclaimers
THE SERVICE IS PROVIDED "AS IS" WITHOUT WARRANTIES OF ANY KIND, TO THE MAXIMUM EXTENT PERMITTED BY LAW.

9. Limitation of Liability
TO THE MAXIMUM EXTENT PERMITTED BY LAW, WE ARE NOT LIABLE FOR INDIRECT, INCIDENTAL, SPECIAL, OR CONSEQUENTIAL DAMAGES ARISING FROM USE OF THE SERVICE.

10. Changes
We may update these Terms. Continued use after changes become effective constitutes acceptance of the revised Terms.

11. Contact
Questions about these Terms: support@softlife.app`,
  privacy: `Last updated: November 1, 2026

1. Introduction
This Privacy Policy explains how Soft Life ("we") collects, uses, and protects personal information when you use our authentication and planning Service.

2. Information We Collect
• Account data: name, email, password hash (we never store plain-text passwords).
• Legal acceptance: Terms/Privacy version and acceptance timestamps.
• Usage and device data needed for security, rate limiting, and product reliability.
• Content you voluntarily upload (e.g. documents for AI assistance).

3. How We Use Information
We use data to create and secure accounts, authenticate sessions, provide AI features, prevent abuse, send transactional emails (such as password resets), and improve the Service.

4. Legal Bases
Where required, we process data based on contract performance, legitimate interests (security and product improvement), and consent (where you opt in).

5. Sharing
We do not sell personal information. We may share data with infrastructure providers under confidentiality obligations, or when required by law.

6. Retention
We retain account data while your account is active and as needed for security, legal, and audit purposes. Password-reset tokens expire automatically and are single-use.

7. Security
We use industry-standard controls including Argon2 password hashing, signed access tokens, hashed refresh/reset tokens, HTTPS where configured, and rate limiting on authentication endpoints.

8. Your Rights
Depending on your location, you may request access, correction, deletion, or export of your personal data. Contact support@softlife.app.

9. Children
The Service is not directed to children under 13 (or higher age where required). We do not knowingly collect such data.

10. International Transfers
Data may be processed in countries where our providers operate, with appropriate safeguards.

11. Changes
We may update this Policy and will revise the version date when we do.

12. Contact
Privacy questions: privacy@softlife.app`,
} as const;
