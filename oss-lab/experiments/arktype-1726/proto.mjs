// Prototype: make the ISO 8601 matcher reject days that don't exist, keeping it a
// pure regex (so toJsonSchema keeps emitting `pattern`).
const original =
	/^([+-]?\d{4}(?!\d{2}\b))((-?)((0[1-9]|1[0-2])(\3([12]\d|0[1-9]|3[01]))?|W([0-4]\d|5[0-3])(-?[1-7])?|(00[1-9]|0[1-9]\d|[12]\d{2}|3([0-5]\d|6[1-6])))(T((([01]\d|2[0-3])((:?)[0-5]\d)?|24:?00)([,.]\d+(?!:))?)?(\17[0-5]\d([,.]\d+)?)?([Zz]|([+-])([01]\d|2[0-3]):?([0-5]\d)?)?)?)?$/

const leapYear = String.raw`(?:\d\d(?:0[48]|[2468][048]|[13579][26])|(?:[02468][048]|[13579][26])00)`
// Month/day with per-month day limits. Group count is kept identical to the
// original (groups 5,6,7 still exist) so later backreferences (\17) are unchanged.
const monthDay = String.raw`(?=(0[1-9]|1[0-2]))(?:(?:0[13578]|1[02])(\3(?:0[1-9]|[12]\d|3[01]))?|(?:0[469]|11)(?:\3(?:0[1-9]|[12]\d|30))?|02(?:\3(0[1-9]|1\d|2[0-9]))?)`
// Feb 29 only in leap years: reject "<non-leap year>-02-29" up front.
const noFeb29InCommonYears = String.raw`(?![+-]?(?!${leapYear})\d{4}-?02-?29)`

const src = original.source
	.replace("^", "^" + noFeb29InCommonYears)
	.replace(String.raw`(0[1-9]|1[0-2])(\3([12]\d|0[1-9]|3[01]))?`, monthDay)
const proposed = new RegExp(src)

const groups = (re) => new RegExp(re.source + "|").exec("").length - 1
console.log("capture groups original/proposed:", groups(original), groups(proposed))
console.log(proposed.source)
export { original, proposed }
