import { describe, expect, it } from 'vitest';
import { formatApiError, formatValidationDetails } from '@/utils/errors';

describe('formatApiError', () => {
  it('surfaces FastAPI detail.error codes', () => {
    expect(
      formatApiError(401, {
        detail: { error: 'Invalid credentials', code: 'INVALID_CREDENTIALS' },
      })
    ).toBe('Invalid email or password');
  });

  it('formats validation details', () => {
    expect(
      formatValidationDetails([
        { msg: 'Value error, Password must include a digit' },
      ])
    ).toBe('Password must include a digit');
  });
});
