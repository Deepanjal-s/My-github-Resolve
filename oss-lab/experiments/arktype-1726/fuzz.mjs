import { original, proposed as protoRe } from "./proto.mjs"
const proposed = globalThis.__SHIPPED__ ?? protoRe

const pad = (n, w) => String(n).padStart(w, "0")
const isLeap = (y) => (y % 4 === 0 && y % 100 !== 0) || y % 400 === 0
const daysIn = (y, m) => [31, isLeap(y) ? 29 : 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31][m - 1]

let checked = 0
const fail = []
const expect = (s, want) => {
	checked++
	if (proposed.test(s) !== want) fail.push(`${s} expected ${want}`)
}

// Every year 0000-9999 x every month x days 00-32 (extended form); a sample of
// years also in basic form, with a time, and with an explicit sign.
for (let y = 0; y <= 9999; y++) {
	for (let m = 1; m <= 12; m++) {
		const Y = pad(y, 4), M = pad(m, 2)
		for (let d = 0; d <= 32; d++) {
			const valid = d >= 1 && d <= daysIn(y, m)
			const D = pad(d, 2)
			expect(`${Y}-${M}-${D}`, valid)
			if (y % 97 === 0 || y % 400 === 0 || y % 100 === 0) {
				expect(`${Y}${M}${D}`, valid)
				expect(`${Y}-${M}-${D}T12:30:00Z`, valid)
				expect(`+${Y}-${M}-${D}`, valid)
				expect(`-${Y}-${M}-${D}`, valid)
			}
		}
		// month-only forms unchanged
		expect(`${Y}-${M}`, original.test(`${Y}-${M}`))
	}
}

// Everything that isn't a calendar date must behave exactly like the original.
const unchanged = [
	"2024", "2024-W01", "2024-W53-7", "2024W017", "2024-001", "2024-366", "2023-366", "2024001", "2023029",
	"2024-02-28T24:00", "2024-01-31T23:59:59.123+05:30", "2024-01-01T10:00:00,5Z", "2024-0101",
	"202401-01", "2024-01-01T10:00:00.000-0800", "2024-1", "24-01-01", "2024-13-01", "2024-01-01T10:0000",
	"2024-01-01T1000:00", "2024-01-01T10:00:00z", "-0001-01-01", "2024-01-01T", "2024-01-01T10", "123456-01-01",
]
for (const s of unchanged) expect(s, original.test(s))

// The cases from the issue.
for (const [s, want] of [["2024-06-31", false], ["2023-02-29", false], ["2024-04-31", false], ["2024-02-29", true], ["2000-02-29", true], ["1900-02-29", false], ["2024-01-32", false]])
	expect(s, want)

const groups = (re) => new RegExp(re.source + "|").exec("").length - 1
console.log(`capture groups ${groups(original)} -> ${groups(proposed)}; checked ${checked} strings; failures ${fail.length}`)
if (fail.length) console.log(fail.slice(0, 20).join("\n"))

// Rough cost: time 200k matches with each regex.
const sample = Array.from({ length: 200_000 }, (_, i) => `${pad(1900 + (i % 300), 4)}-${pad((i % 12) + 1, 2)}-${pad((i % 31) + 1, 2)}T10:20:30.123Z`)
for (const [name, re] of [["original", original], ["proposed", proposed]]) {
	const t = performance.now()
	let n = 0
	for (const s of sample) n += re.test(s)
	console.log(name, `${(performance.now() - t).toFixed(1)} ms`, n)
}
