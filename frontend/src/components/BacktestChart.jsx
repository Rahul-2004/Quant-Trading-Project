import React from 'react';
import {
    CartesianGrid,
    Legend,
    Line,
    LineChart,
    ResponsiveContainer,
    Tooltip,
    XAxis,
    YAxis,
} from 'recharts';

const parseYear = (timestamp) => {
    if (!timestamp) return null;
    const direct = Number.parseInt(String(timestamp).slice(0, 4), 10);
    if (Number.isFinite(direct)) return direct;

    const date = new Date(timestamp);
    if (Number.isNaN(date.getTime())) return null;
    return date.getUTCFullYear();
};

const parseDate = (timestamp) => {
    if (!timestamp) return null;
    const date = new Date(timestamp);
    if (Number.isNaN(date.getTime())) return null;
    return date;
};

const normalizeAllocation = (allocation) => {
    const btc = Number(allocation?.btc ?? 0);
    const gold = Number(allocation?.gold ?? 0);
    const silver = Number(allocation?.silver ?? 0);
    return {
        btc: Number.isFinite(btc) ? btc : 0,
        gold: Number.isFinite(gold) ? gold : 0,
        silver: Number.isFinite(silver) ? silver : 0,
    };
};

const allocationLabel = (allocation) => `B${allocation.btc.toFixed(0)} G${allocation.gold.toFixed(0)} S${allocation.silver.toFixed(0)}`;

const parseMonthKey = (timestamp) => {
    const date = parseDate(timestamp);
    if (!date) return null;
    const month = String(date.getUTCMonth() + 1).padStart(2, '0');
    return `${date.getUTCFullYear()}-${month}`;
};

const MAX_EXP_INPUT = 700;

const growthFromIndex = (startValue, endValue) => {
    if (!Number.isFinite(startValue) || !Number.isFinite(endValue) || startValue <= 0) return Number.NaN;
    return endValue / startValue;
};

const growthFromLogCurve = (startValue, endValue) => {
    if (!Number.isFinite(startValue) || !Number.isFinite(endValue)) return Number.NaN;
    const delta = endValue - startValue;
    if (delta > MAX_EXP_INPUT) return Number.POSITIVE_INFINITY;
    if (delta < -MAX_EXP_INPUT) return 0;
    return Math.exp(delta);
};

const scoreCandidateSeries = (rows, valueKey) => {
    if (!rows?.length) return Number.POSITIVE_INFINITY;
    let score = 0;
    for (const row of rows) {
        const value = Number(row?.[valueKey]);
        if (!Number.isFinite(value)) {
            score += 1e6;
            continue;
        }
        const absValue = Math.abs(value);
        if (absValue > 1000) score += absValue;
        else score += absValue * 0.01;
    }
    return score;
};

const pickBetterSeries = (indexRows, logRows, valueKey) => {
    const indexScore = scoreCandidateSeries(indexRows, valueKey);
    const logScore = scoreCandidateSeries(logRows, valueKey);
    return logScore < indexScore ? logRows : indexRows;
};

const sanitizeMetricRows = (rows, valueKey) => rows.map((row) => {
    const metric = Number(row?.[valueKey]);
    if (Number.isFinite(metric)) return row;
    return {
        ...row,
        [valueKey]: 0,
        metricValue: 0,
    };
});

const AllocationDot = ({ cx, cy, payload, stroke, onYearSelect, selectedYear }) => {
    if (typeof cx !== 'number' || typeof cy !== 'number' || !payload?.allocation) return null;
    const isSelected = selectedYear === payload?.year;
    const handleClick = () => {
        if (typeof onYearSelect === 'function') {
            onYearSelect(payload.year, payload.allocation);
        }
    };
    return (
        <g onClick={handleClick} style={{ cursor: onYearSelect ? 'pointer' : 'default' }}>
            <circle
                cx={cx}
                cy={cy}
                r={isSelected ? 5.5 : 4}
                fill={stroke || '#22d3ee'}
                stroke={isSelected ? '#facc15' : '#0b1220'}
                strokeWidth={isSelected ? 2 : 1}
            />
            <text x={cx + 6} y={cy - 6} fill="#9ca3af" fontSize={10}>
                {allocationLabel(payload.allocation)}
            </text>
        </g>
    );
};

const CustomTooltip = ({ active, payload, label }) => {
    if (!active || !payload?.length) return null;
    const point = payload[0]?.payload;
    const allocation = normalizeAllocation(point?.allocation);

    return (
        <div className="rounded-md border border-gray-700 bg-gray-900 px-3 py-2 text-xs text-gray-200 shadow-xl">
            <div className="font-semibold text-gray-100">{point?.labelPrefix || 'Period'} {label}</div>
            <div className="mt-1">{point?.metricLabel || 'Return'}: {Number(point?.metricValue ?? 0).toFixed(2)}%</div>
            <div className="mt-1">Allocation: {allocationLabel(allocation)}%</div>
        </div>
    );
};

const buildYearlyYoyCagr = (equityData, timestamps, allocations) => {
    if (!equityData?.length) return [];

    const yearlyStats = new Map();
    const rows = Math.min(equityData.length, timestamps?.length || 0);
    for (let index = 0; index < rows; index += 1) {
        const timestamp = timestamps[index];
        const year = parseYear(timestamp);
        const date = parseDate(timestamp);
        if (!year || !date) continue;

        const rawValue = Number(equityData[index] ?? Number.NaN);
        if (!Number.isFinite(rawValue)) continue;
        const allocation = normalizeAllocation(allocations?.[index]);
        const existing = yearlyStats.get(year);
        if (!existing) {
            yearlyStats.set(year, {
                year,
                firstDate: date,
                firstValue: rawValue,
                lastDate: date,
                lastValue: rawValue,
                allocBtc: allocation.btc,
                allocGold: allocation.gold,
                allocSilver: allocation.silver,
                allocCount: 1,
            });
            continue;
        }

        existing.lastDate = date;
        existing.lastValue = rawValue;
        existing.allocBtc += allocation.btc;
        existing.allocGold += allocation.gold;
        existing.allocSilver += allocation.silver;
        existing.allocCount += 1;
    }

    const ordered = Array.from(yearlyStats.values())
        .sort((a, b) => a.year - b.year);

    if (!ordered.length) return [];

    const MS_PER_DAY = 24 * 60 * 60 * 1000;
    const annualizationDays = 365.25;
    const MIN_FULL_YEAR_COVERAGE_DAYS = 300;

    const buildFullYearRows = (growthFn) => ordered
        .map((current) => {
            const days = (current.lastDate.getTime() - current.firstDate.getTime()) / MS_PER_DAY;
            if (days < MIN_FULL_YEAR_COVERAGE_DAYS) return null;

            const growth = growthFn(current.firstValue, current.lastValue);
            const yoyCagrPct = (Number.isFinite(growth) && growth > 0 && days > 0)
                ? ((growth ** (annualizationDays / days)) - 1) * 100
                : Number.NaN;

            const count = Math.max(current.allocCount, 1);
            return {
                year: current.year,
                yoy_cagr_pct: yoyCagrPct,
                metricValue: yoyCagrPct,
                metricLabel: 'YoY CAGR',
                labelPrefix: 'Year',
                allocation: {
                    btc: current.allocBtc / count,
                    gold: current.allocGold / count,
                    silver: current.allocSilver / count,
                },
            };
        })
        .filter(Boolean);

    const fullYearRowsIndex = buildFullYearRows(growthFromIndex);
    const fullYearRowsLog = buildFullYearRows(growthFromLogCurve);
    const selectedFullYearRows = pickBetterSeries(fullYearRowsIndex, fullYearRowsLog, 'yoy_cagr_pct');
    if (selectedFullYearRows.length > 0) return sanitizeMetricRows(selectedFullYearRows, 'yoy_cagr_pct');

    // Fallback when no full-year coverage exists: compute one annualized point over the whole span.
    const firstYearPoint = ordered[0];
    const lastYearPoint = ordered[ordered.length - 1];
    const spanDays = Math.max((lastYearPoint.lastDate.getTime() - firstYearPoint.firstDate.getTime()) / MS_PER_DAY, 1);
    const indexGrowth = growthFromIndex(firstYearPoint.firstValue, lastYearPoint.lastValue);
    const logGrowth = growthFromLogCurve(firstYearPoint.firstValue, lastYearPoint.lastValue);
    const indexSpanCagrPct = (Number.isFinite(indexGrowth) && indexGrowth > 0)
        ? ((indexGrowth ** (annualizationDays / spanDays)) - 1) * 100
        : Number.NaN;
    const logSpanCagrPct = (Number.isFinite(logGrowth) && logGrowth > 0)
        ? ((logGrowth ** (annualizationDays / spanDays)) - 1) * 100
        : Number.NaN;
    const fallbackRows = [
        {
            year: lastYearPoint.year,
            yoy_cagr_pct: indexSpanCagrPct,
            metricValue: indexSpanCagrPct,
            metricLabel: 'YoY CAGR',
            labelPrefix: 'Year',
            allocation: {
                btc: lastYearPoint.allocBtc / Math.max(lastYearPoint.allocCount, 1),
                gold: lastYearPoint.allocGold / Math.max(lastYearPoint.allocCount, 1),
                silver: lastYearPoint.allocSilver / Math.max(lastYearPoint.allocCount, 1),
            },
        },
    ];
    const fallbackLogRows = [
        {
            ...fallbackRows[0],
            yoy_cagr_pct: logSpanCagrPct,
            metricValue: logSpanCagrPct,
        },
    ];
    return sanitizeMetricRows(pickBetterSeries(fallbackRows, fallbackLogRows, 'yoy_cagr_pct'), 'yoy_cagr_pct');
};

const buildMonthlyReturnPct = (equityData, timestamps, allocations) => {
    if (!equityData?.length) return [];

    const monthlyStats = new Map();
    const rows = Math.min(equityData.length, timestamps?.length || 0);
    for (let index = 0; index < rows; index += 1) {
        const month = parseMonthKey(timestamps[index]);
        const year = parseYear(timestamps[index]);
        if (!month || !year) continue;

        const rawValue = Number(equityData[index] ?? Number.NaN);
        if (!Number.isFinite(rawValue)) continue;
        const allocation = normalizeAllocation(allocations?.[index]);
        const existing = monthlyStats.get(month);
        if (!existing) {
            monthlyStats.set(month, {
                month,
                year,
                firstValue: rawValue,
                lastValue: rawValue,
                allocBtc: allocation.btc,
                allocGold: allocation.gold,
                allocSilver: allocation.silver,
                allocCount: 1,
            });
            continue;
        }

        existing.lastValue = rawValue;
        existing.allocBtc += allocation.btc;
        existing.allocGold += allocation.gold;
        existing.allocSilver += allocation.silver;
        existing.allocCount += 1;
    }

    const ordered = Array.from(monthlyStats.values()).sort((a, b) => a.month.localeCompare(b.month));
    if (!ordered.length) return [];

    const buildRows = (growthFn) => {
        let priorEnd = ordered[0].firstValue;
        return ordered.map((item) => {
            const growth = growthFn(priorEnd, item.lastValue);
            const monthlyReturnPct = (Number.isFinite(growth) && growth > 0)
                ? (growth - 1) * 100
                : Number.NaN;
            priorEnd = item.lastValue;
            const count = Math.max(item.allocCount, 1);
            return {
                month: item.month,
                year: item.year,
                return_pct: monthlyReturnPct,
                metricValue: monthlyReturnPct,
                metricLabel: 'Monthly Return',
                labelPrefix: 'Month',
                allocation: {
                    btc: item.allocBtc / count,
                    gold: item.allocGold / count,
                    silver: item.allocSilver / count,
                },
            };
        });
    };

    const monthlyRowsIndex = buildRows(growthFromIndex);
    const monthlyRowsLog = buildRows(growthFromLogCurve);
    return sanitizeMetricRows(pickBetterSeries(monthlyRowsIndex, monthlyRowsLog, 'return_pct'), 'return_pct');
};

const BacktestChart = ({
    data,
    timestamps,
    allocations,
    onYearSelect,
    selectedYear,
    mode = 'yearly_yoy_cagr',
}) => {
    const chartData = mode === 'monthly_return_pct'
        ? buildMonthlyReturnPct(data, timestamps, allocations)
        : buildYearlyYoyCagr(data, timestamps, allocations);
    if (chartData.length === 0) return null;

    const valueKey = mode === 'monthly_return_pct' ? 'return_pct' : 'yoy_cagr_pct';
    const xDataKey = mode === 'monthly_return_pct' ? 'month' : 'year';
    const lineName = mode === 'monthly_return_pct' ? 'Monthly Return %' : 'YoY CAGR %';

    const values = chartData.map((row) => Number(row[valueKey] ?? 0)).filter((value) => Number.isFinite(value));
    const minValue = Math.min(...values);
    const maxValue = Math.max(...values);
    const range = maxValue - minValue;
    const yPadding = range > 0 ? range * 0.12 : Math.max(Math.abs(maxValue) * 0.2, 5);
    const yDomain = [minValue - yPadding, maxValue + yPadding];

    return (
        <div className="h-80 w-full">
            <ResponsiveContainer width="100%" height="100%">
                <LineChart data={chartData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                    <XAxis dataKey={xDataKey} stroke="#9ca3af" interval="preserveStartEnd" minTickGap={28} />
                    <YAxis domain={yDomain} stroke="#9ca3af" tickFormatter={(value) => `${Number(value).toFixed(0)}%`} tickCount={6} />
                    <Tooltip content={<CustomTooltip />} />
                    <Legend />
                    <Line
                        type="monotone"
                        dataKey={valueKey}
                        stroke="#22d3ee"
                        strokeWidth={2}
                        dot={(dotProps) => (
                            <AllocationDot
                                {...dotProps}
                                onYearSelect={onYearSelect}
                                selectedYear={selectedYear}
                            />
                        )}
                        name={lineName}
                    />
                </LineChart>
            </ResponsiveContainer>
        </div>
    );
};

export default BacktestChart;
