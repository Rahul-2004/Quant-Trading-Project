export const shortStrategy = (value = '') => value.replace(/^\d+_/, '').replace(/_/g, ' ').slice(0, 28);

export const round = (value, digits = 2) => Number(value ?? 0).toFixed(digits);

export const percent = (value, digits = 2) => `${round(value, digits)}%`;

export const currency = (value) =>
    new Intl.NumberFormat('en-US', {
        style: 'currency',
        currency: 'USD',
        maximumFractionDigits: 2,
    }).format(Number(value ?? 0));
