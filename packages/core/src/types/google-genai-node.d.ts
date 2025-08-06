/**
 * TypeScript Declaration for Google GenAI Node.js Import
 *
 * This declaration file bridges the gap between the platform-specific import
 * '@google/genai/node' and the main package types. The @google/genai package
 * may not include complete TypeScript definitions for the /node subpath export.
 *
 * This tells TypeScript: "when importing from '@google/genai/node',
 * use the same types as '@google/genai'" ensuring proper type safety and IDE support.
 *
 * Can be removed if/when the upstream package provides complete type definitions
 * for the platform-specific exports.
 */
declare module '@google/genai/node' {
    export * from '@google/genai';
}
