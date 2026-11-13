// Re-run fuzz.mjs's checks against the regex exactly as committed upstream.
import { readFileSync } from "node:fs"
const file = process.argv[2]
const src = readFileSync(file, "utf8").match(/export const iso8601Matcher =\s*\/(.+)\/\n/)[1]
globalThis.__SHIPPED__ = new RegExp(src)
await import("./fuzz.mjs")
