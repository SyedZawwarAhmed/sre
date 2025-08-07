import os from 'os';
import path from 'path';
import EventEmitter from 'events';
import fs from 'fs';

/**
 * Google GenAI Platform-Specific Import
 *
 * Using '@google/genai/node' instead of '@google/genai' because:
 * - The generic '@google/genai' import throws an error requiring platform-specific imports
 * - '/node' provides Node.js-optimized implementation with server-side features:
 *   * File system access for uploads
 *   * Server environment optimizations
 *   * Full API feature support
 * - '/web' would be for browser environments (excludes Node.js APIs)
 *
 * This ensures we get the correct implementation for our server-side environment.
 */
import { GoogleGenAI } from '@google/genai/node';

import { JSON_RESPONSE_INSTRUCTION, BUILT_IN_MODEL_PREFIX } from '@sre/constants';
import { BinaryInput } from '@sre/helpers/BinaryInput.helper';
import { AccessCandidate } from '@sre/Security/AccessControl/AccessCandidate.class';
import { uid } from '@sre/utils';

import { processWithConcurrencyLimit } from '@sre/utils';

import {
    TLLMMessageBlock,
    ToolData,
    TLLMMessageRole,
    TLLMToolResultMessageBlock,
    APIKeySource,
    TLLMEvent,
    BasicCredentials,
    ILLMRequestFuncParams,
    TLLMChatResponse,
    TGoogleAIRequestBody,
    ILLMRequestContext,
    TLLMPreparedParams,
} from '@sre/types/LLM.types';
import { LLMHelper } from '@sre/LLMManager/LLM.helper';

import { SystemEvents } from '@sre/Core/SystemEvents';
import { SUPPORTED_MIME_TYPES_MAP } from '@sre/constants';

import { LLMConnector } from '../LLMConnector';

const MODELS_SUPPORT_SYSTEM_INSTRUCTION = [
    'gemini-1.5-pro-exp-0801',
    'gemini-1.5-pro-latest',
    'gemini-1.5-pro-latest',
    'gemini-1.5-pro',
    'gemini-1.5-pro-001',
    'gemini-1.5-flash-latest',
    'gemini-1.5-flash-001',
    'gemini-1.5-flash',
];
const MODELS_SUPPORT_JSON_RESPONSE = MODELS_SUPPORT_SYSTEM_INSTRUCTION;

// Supported file MIME types for Google AI's Gemini models
const VALID_MIME_TYPES = [
    ...SUPPORTED_MIME_TYPES_MAP.GoogleAI.image,
    ...SUPPORTED_MIME_TYPES_MAP.GoogleAI.audio,
    ...SUPPORTED_MIME_TYPES_MAP.GoogleAI.video,
    ...SUPPORTED_MIME_TYPES_MAP.GoogleAI.document,
];

// Usage metadata type for the new SDK
type UsageMetadataWithThoughtsToken = {
    promptTokenCount?: number;
    candidatesTokenCount?: number;
    cachedContentTokenCount?: number;
    thoughtsTokenCount?: number;
    promptTokensDetails?: Array<{ modality: string; tokenCount: number }>;
};

export class GoogleAIConnector extends LLMConnector {
    public name = 'LLM:GoogleAI';

    private validMimeTypes = {
        all: VALID_MIME_TYPES,
        image: SUPPORTED_MIME_TYPES_MAP.GoogleAI.image,
    };

    private async getClient(params: ILLMRequestContext): Promise<GoogleGenAI> {
        const apiKey = (params.credentials as BasicCredentials)?.apiKey;

        if (!apiKey) throw new Error('Please provide an API key for Google AI');

        return new GoogleGenAI({ apiKey });
    }

    protected async request({ acRequest, body, context }: ILLMRequestFuncParams): Promise<TLLMChatResponse> {
        try {
            const genAI = await this.getClient(context);

            // Extract messages and structure payload for Google AI API
            const { messages: contents, ...requestPayload } = body;

            const result = await genAI.models.generateContent({ ...requestPayload, contents });

            const content = result.text;
            const finishReason = result.candidates?.[0]?.finishReason || 'stop';
            const usage = result.usageMetadata as UsageMetadataWithThoughtsToken;
            this.reportUsage(usage, {
                modelEntryName: context.modelEntryName,
                keySource: context.isUserKey ? APIKeySource.User : APIKeySource.Smyth,
                agentId: context.agentId,
                teamId: context.teamId,
            });

            const toolCalls = result.candidates?.[0]?.content?.parts?.filter((part) => part.functionCall);

            let toolsData: ToolData[] = [];
            let useTool = false;

            if (toolCalls && toolCalls.length > 0) {
                toolsData = toolCalls.map((toolCall, index) => ({
                    index,
                    id: `tool-${index}`,
                    type: 'function',
                    name: toolCall.functionCall.name,
                    arguments: JSON.stringify(toolCall.functionCall.args),
                    role: TLLMMessageRole.Assistant,
                }));
                useTool = true;
            }

            // Handle thinking summary if available
            let thinkingSummary = null;
            if ((result as any).thoughtSummary) {
                thinkingSummary = (result as any).thoughtSummary;
            }

            // Handle images in response
            const images: any[] = [];
            if (result.candidates) {
                for (const candidate of result.candidates) {
                    if (candidate.content?.parts) {
                        for (const part of candidate.content.parts) {
                            if (part.inlineData && part.inlineData.mimeType?.startsWith('image/')) {
                                images.push({
                                    mimeType: part.inlineData.mimeType,
                                    data: part.inlineData.data,
                                });
                            }
                        }
                    }
                }
            }

            return {
                content,
                finishReason,
                useTool,
                toolsData,
                message: { content, role: 'assistant' },
                usage,
                ...(thinkingSummary && { thinkingSummary }),
                ...(images.length > 0 && { images }),
            };
        } catch (error: any) {
            throw error;
        }
    }

    protected async streamRequest({ acRequest, body, context }: ILLMRequestFuncParams): Promise<EventEmitter> {
        const emitter = new EventEmitter();

        const genAI = await this.getClient(context);

        // Extract messages and structure payload for Google AI API
        const { messages: contents, ...requestPayload } = body;

        const result = await genAI.models.generateContentStream({ ...requestPayload, contents });

        let toolsData: ToolData[] = [];
        let usage: UsageMetadataWithThoughtsToken;

        // Process stream asynchronously while as we need to return emitter immediately
        (async () => {
            for await (const chunk of result) {
                // Handle text content
                if (chunk.text) {
                    emitter.emit('content', chunk.text);
                }

                // Handle thinking content if available
                if ((chunk as any).thoughtSummary) {
                    emitter.emit('thinking', (chunk as any).thoughtSummary);
                }

                if (chunk.candidates?.[0]?.content?.parts) {
                    const parts = chunk.candidates[0].content.parts;

                    // Handle tool calls
                    const toolCalls = parts.filter((part) => part.functionCall);
                    if (toolCalls.length > 0) {
                        toolsData = toolCalls.map((toolCall, index) => ({
                            index,
                            id: `tool-${index}`,
                            type: 'function',
                            name: toolCall.functionCall.name,
                            arguments: JSON.stringify(toolCall.functionCall.args),
                            role: TLLMMessageRole.Assistant,
                        }));
                        emitter.emit(TLLMEvent.ToolInfo, toolsData);
                    }

                    // Handle image responses
                    const imageParts = parts.filter((part) => part.inlineData?.mimeType?.startsWith('image/'));
                    if (imageParts.length > 0) {
                        emitter.emit(
                            'images',
                            imageParts.map((part) => ({
                                mimeType: part.inlineData.mimeType,
                                data: part.inlineData.data,
                            }))
                        );
                    }
                }

                // the same usage is sent on each emit. IMPORTANT: google does not send usage for each chunk but
                // rather just sends the same usage for the entire request.
                // notice that the output tokens are only sent in the last chunk usage metadata.
                // so we will just update a var to hold the latest usage and report it when the stream ends.
                // e.g emit1: { input_tokens: 500, output_tokens: undefined } -> same input_tokens
                // e.g emit2: { input_tokens: 500, output_tokens: undefined } -> same input_tokens
                // e.g emit3: { input_tokens: 500, output_tokens: 10 } -> same input_tokens, new output_tokens in the last chunk
                if (chunk?.usageMetadata) {
                    usage = chunk.usageMetadata as UsageMetadataWithThoughtsToken;
                }
            }

            if (usage) {
                this.reportUsage(usage, {
                    modelEntryName: context.modelEntryName,
                    keySource: context.isUserKey ? APIKeySource.User : APIKeySource.Smyth,
                    agentId: context.agentId,
                    teamId: context.teamId,
                });
            }

            setTimeout(() => {
                emitter.emit('end', toolsData);
            }, 100);
        })();

        return emitter;
    }
    // #region Image Generation, will be moved to a different subsystem/service
    protected async imageGenRequest({ body, context }: ILLMRequestFuncParams): Promise<any> {
        try {
            const genAI = await this.getClient(context);

            // Use Gemini 2.0 Flash for image generation or fallback to Imagen
            const model = body.model || 'gemini-2.0-flash-preview-image-generation';

            // Build the request with proper response modalities
            const config: any = {
                responseModalities: ['TEXT', 'IMAGE'],
            };

            // Add image-specific configuration if available
            if (body.n) config.numberOfImages = body.n;
            if (body.aspect_ratio || body.size) {
                config.aspectRatio = body.aspect_ratio || body.size || '1:1';
            }
            if (body.person_generation) {
                config.personGeneration = body.person_generation;
            }

            const requestPayload = {
                model,
                contents: body.prompt,
                config,
            };

            // Generate content with image modality
            const response = await genAI.models.generateContent(requestPayload);

            // Extract images from the response
            const images: any[] = [];
            if (response.candidates) {
                for (const candidate of response.candidates) {
                    if (candidate.content?.parts) {
                        for (const part of candidate.content.parts) {
                            if (part.inlineData && part.inlineData.mimeType?.startsWith('image/')) {
                                images.push({
                                    url: `data:${part.inlineData.mimeType};base64,${part.inlineData.data}`,
                                    b64_json: part.inlineData.data,
                                    revised_prompt: body.prompt,
                                });
                            }
                        }
                    }
                }
            }

            // Transform the response to match OpenAI format for compatibility
            return {
                created: Math.floor(Date.now() / 1000),
                data: images,
            };
        } catch (error: any) {
            throw error;
        }
    }

    protected async imageEditRequest({ body, context }: ILLMRequestFuncParams): Promise<any> {
        throw new Error('Image editing is not supported for Google AI. Imagen models only support image generation.');
    }

    protected async reqBodyAdapter(params: TLLMPreparedParams): Promise<TGoogleAIRequestBody> {
        const model = params?.model;

        // Check if this is an image generation request based on capabilities
        if (params?.capabilities?.imageGeneration) {
            return this.prepareBodyForImageGenRequest(params) as any;
        }

        const messagesResult = await this.prepareMessages(params);

        let body: any = {
            model: model as string,
            messages: Array.isArray(messagesResult) ? messagesResult : messagesResult.contents,
        };

        // Preserve tools configuration if it exists (from prepareMessagesWithTools)
        if (!Array.isArray(messagesResult)) {
            if (messagesResult.tools) body.tools = messagesResult.tools;
            if (messagesResult.toolConfig) body.toolConfig = messagesResult.toolConfig;
            if (messagesResult.systemInstruction) body.systemInstruction = messagesResult.systemInstruction;
        }

        const responseFormat = params?.responseFormat || '';
        let systemInstruction = '';

        // Handle system instruction from params (only if not already set from tools flow)
        if (!body.systemInstruction) {
            if ((params as any).systemInstruction) {
                systemInstruction += (params as any).systemInstruction + '\n';
            }

            if (responseFormat === 'json') {
                systemInstruction += JSON_RESPONSE_INSTRUCTION;
            }
        }

        const config: any = {};

        // Basic generation parameters
        if (params.maxTokens !== undefined) config.maxOutputTokens = params.maxTokens;
        if (params.temperature !== undefined) config.temperature = params.temperature;
        if (params.topP !== undefined) config.topP = params.topP;
        if (params.topK !== undefined) config.topK = params.topK;
        if (params.stopSequences?.length) config.stopSequences = params.stopSequences;

        // Response format
        if (responseFormat === 'json' && MODELS_SUPPORT_JSON_RESPONSE.includes(model as string)) {
            config.responseMimeType = 'application/json';
        }

        // Thinking support - auto-enable for reasoning-capable models
        if ((params as any).thinkingBudget !== undefined) {
            config.thinkingBudget = (params as any).thinkingBudget;
        } else if (params.capabilities?.reasoning) {
            // Set thinkingBudget to -1 (auto) for models with reasoning capabilities
            config.thinkingBudget = -1;
        }

        if ((params as any).includeThoughts !== undefined) {
            config.includeThoughts = (params as any).includeThoughts;
        } else if (params.capabilities?.reasoning) {
            // Auto-enable includeThoughts for reasoning-capable models
            config.includeThoughts = true;
        }

        // Response modalities for multimodal responses
        if ((params as any).responseModalities) {
            config.responseModalities = (params as any).responseModalities;
        }

        // Audio processing configuration
        if ((params as any).audioConfig) {
            config.audioConfig = (params as any).audioConfig;
        }

        // Video processing configuration
        if ((params as any).videoConfig) {
            config.videoConfig = (params as any).videoConfig;
        }

        if (systemInstruction.trim() && !body.systemInstruction) {
            body.systemInstruction = systemInstruction.trim();
        }

        // Structure body to match Google AI API format directly
        if (Object.keys(config).length > 0) {
            body.config = config; // Use 'config' instead of 'generationConfig' for API
        }

        return body;
    }

    protected reportUsage(
        usage: UsageMetadataWithThoughtsToken,
        metadata: { modelEntryName: string; keySource: APIKeySource; agentId: string; teamId: string }
    ) {
        const modelEntryName = metadata.modelEntryName;
        let tier = '';

        const tierThresholds = {
            'gemini-1.5-pro': 128_000,
            'gemini-2.5-pro': 200_000,
        };

        const textInputTokens =
            usage?.['promptTokensDetails']?.find((detail) => detail.modality === 'TEXT')?.tokenCount || usage?.promptTokenCount || 0;
        const audioInputTokens = usage?.['promptTokensDetails']?.find((detail) => detail.modality === 'AUDIO')?.tokenCount || 0;

        // Find matching model and set tier based on threshold
        const modelWithTier = Object.keys(tierThresholds).find((model) => modelEntryName.includes(model));
        if (modelWithTier) {
            tier = textInputTokens < tierThresholds[modelWithTier] ? 'tier1' : 'tier2';
        }

        // #endregion

        // SmythOS (built-in) models have a prefix, so we need to remove it to get the model name
        const modelName = metadata.modelEntryName.replace(BUILT_IN_MODEL_PREFIX, '');

        const usageData = {
            sourceId: `llm:${modelName}`,
            input_tokens: textInputTokens,
            output_tokens: usage.candidatesTokenCount + (usage.thoughtsTokenCount || 0),
            input_tokens_audio: audioInputTokens,
            input_tokens_cache_read: usage.cachedContentTokenCount || 0,
            input_tokens_cache_write: 0,
            keySource: metadata.keySource,
            agentId: metadata.agentId,
            teamId: metadata.teamId,
            tier,
        };
        SystemEvents.emit('USAGE:LLM', usageData);

        return usageData;
    }

    public formatToolsConfig({ toolDefinitions, toolChoice = 'auto' }) {
        const tools = toolDefinitions.map((tool) => {
            const { name, description, properties, requiredFields } = tool;

            // Ensure the function name is valid
            const validName = this.sanitizeFunctionName(name);

            // Ensure properties are non-empty for OBJECT type
            const validProperties = properties && Object.keys(properties).length > 0 ? properties : { dummy: { type: 'string' } };

            return {
                functionDeclarations: [
                    {
                        name: validName,
                        description: description || '',
                        parameters: {
                            type: 'OBJECT',
                            properties: validProperties,
                            required: requiredFields || [],
                        },
                    },
                ],
            };
        });

        return {
            tools,
            toolChoice: {
                type: toolChoice,
            },
        };
    }

    public transformToolMessageBlocks({
        messageBlock,
        toolsData,
    }: {
        messageBlock: TLLMMessageBlock;
        toolsData: ToolData[];
    }): TLLMToolResultMessageBlock[] {
        const messageBlocks: TLLMToolResultMessageBlock[] = [];

        // Add the assistant's message with function calls if present
        if (messageBlock) {
            const parts = [];

            // Add text content if exists
            if (typeof messageBlock.content === 'string' && messageBlock.content.trim()) {
                parts.push({ text: messageBlock.content });
            } else if (Array.isArray(messageBlock.content)) {
                parts.push(...messageBlock.content);
            }

            // Add function calls from the assistant message
            if (messageBlock.parts) {
                const functionCalls = messageBlock.parts.filter((part) => part.functionCall);
                if (functionCalls.length > 0) {
                    parts.push(
                        ...functionCalls.map((call) => ({
                            functionCall: {
                                name: call.functionCall.name,
                                args: typeof call.functionCall.args === 'string' ? JSON.parse(call.functionCall.args) : call.functionCall.args,
                            },
                        }))
                    );
                }
            }

            // Add function calls from toolsData if not already in parts
            if (toolsData && toolsData.length > 0 && !parts.some((part) => part.functionCall)) {
                parts.push(
                    ...toolsData.map((tool) => ({
                        functionCall: {
                            name: tool.name,
                            args: typeof tool.arguments === 'string' ? JSON.parse(tool.arguments) : tool.arguments,
                        },
                    }))
                );
            }

            // Only add the message block if it has content
            if (parts.length > 0) {
                messageBlocks.push({
                    role: TLLMMessageRole.Model, // Use 'model' role for assistant messages in Gemini
                    parts,
                });
            }
        }

        // Transform tool results into proper function responses for Gemini
        const transformedToolsData = toolsData
            .filter((toolData) => toolData.result !== undefined) // Only process tools with results
            .map((toolData): TLLMToolResultMessageBlock => {
                // Parse the result properly
                let functionResult: any;
                try {
                    functionResult = typeof toolData.result === 'string' ? JSON.parse(toolData.result) : toolData.result;
                } catch {
                    // If parsing fails, use the result as-is
                    functionResult = toolData.result;
                }

                return {
                    role: TLLMMessageRole.User, // Function responses should be from 'user' role in Gemini
                    parts: [
                        {
                            functionResponse: {
                                name: toolData.name,
                                response: functionResult, // Directly use the result as per Google's documentation
                            },
                        },
                    ],
                };
            });

        return [...messageBlocks, ...transformedToolsData];
    }

    public getConsistentMessages(messages: TLLMMessageBlock[]): TLLMMessageBlock[] {
        const _messages = LLMHelper.removeDuplicateUserMessages(messages);

        const processedMessages = _messages.map((message) => {
            const _message = { ...message };

            // Map roles to valid Google AI roles
            switch (_message.role) {
                case TLLMMessageRole.Assistant:
                case TLLMMessageRole.System:
                    _message.role = TLLMMessageRole.Model;
                    break;
                case TLLMMessageRole.User:
                case TLLMMessageRole.Function:
                    // User and Function roles are valid for Gemini
                    _message.role = TLLMMessageRole.User;
                    break;
                default:
                    _message.role = TLLMMessageRole.User; // Default to user for unknown roles
            }

            // Handle messages that already have parts (including function calls/responses)
            if (_message?.parts && Array.isArray(_message.parts)) {
                // Process parts to ensure proper structure
                _message.parts = _message.parts.map((part: any) => {
                    // Handle function calls
                    if (part.functionCall) {
                        return {
                            functionCall: {
                                name: part.functionCall.name,
                                args: typeof part.functionCall.args === 'string' ? JSON.parse(part.functionCall.args) : part.functionCall.args || {},
                            },
                        };
                    }
                    // Handle function responses
                    if (part.functionResponse) {
                        return {
                            functionResponse: {
                                name: part.functionResponse.name,
                                response: part.functionResponse.response,
                            },
                        };
                    }
                    // Handle regular text
                    return part;
                });
                return _message;
            }

            // Handle text content for regular messages
            let textContent = '';
            if (_message?.parts) {
                textContent = _message.parts.map((textBlock: any) => textBlock?.text || '...').join(' ');
            } else if (Array.isArray(_message?.content)) {
                textContent = _message.content.map((textBlock: any) => textBlock?.text || '...').join(' ');
            } else if (_message?.content) {
                textContent = (_message.content as string) || '...';
            }

            _message.parts = [{ text: textContent || '...' }];
            delete _message.content; // Remove content to avoid error

            return _message;
        });

        // Apply function call response order enforcement for all message flows
        return this.ensureFunctionCallResponseOrder(processedMessages);
    }

    private ensureFunctionCallResponseOrder(messages: TLLMMessageBlock[]): TLLMMessageBlock[] {
        const processedMessages: TLLMMessageBlock[] = [];

        for (let i = 0; i < messages.length; i++) {
            const message = messages[i];

            // Check for consecutive model messages (assistant/model role)
            if (message.role === TLLMMessageRole.Model && processedMessages.length > 0) {
                const lastMessage = processedMessages[processedMessages.length - 1];

                // If the last message was also a model message, we need to insert a user turn
                if (lastMessage.role === TLLMMessageRole.Model) {
                    // Insert a minimal user acknowledgment to maintain proper turn order
                    processedMessages.push({
                        role: TLLMMessageRole.User,
                        parts: [{ text: 'Continue.' }],
                    });
                }
            }

            processedMessages.push(message);
        }

        // Ensure conversation ends with proper turn order
        if (processedMessages.length > 1) {
            const lastMessage = processedMessages[processedMessages.length - 1];
            const secondLastMessage = processedMessages[processedMessages.length - 2];

            // If we have a model message with function calls at the end without a response,
            // this is OK as it's a new function call request
            if (lastMessage.role === TLLMMessageRole.Model && lastMessage.parts) {
                const hasFunctionCall = lastMessage.parts.some((part: any) => part.functionCall);
                if (hasFunctionCall && secondLastMessage.role !== TLLMMessageRole.User) {
                    // Insert a user turn before the function call
                    processedMessages.splice(-1, 0, {
                        role: TLLMMessageRole.User,
                        parts: [{ text: 'Please proceed.' }],
                    });
                }
            }
        }

        return processedMessages;
    }

    private async prepareMessages(params: TLLMPreparedParams): Promise<any[] | any> {
        const files: BinaryInput[] = params?.files || [];

        if (files.length > 0) {
            return await this.prepareMessagesWithFiles(params);
        } else if (params?.toolsConfig?.tools?.length > 0) {
            return await this.prepareMessagesWithTools(params);
        } else {
            return await this.prepareMessagesWithTextQuery(params);
        }
    }

    private async prepareMessagesWithFiles(params: TLLMPreparedParams): Promise<any[]> {
        const model = params.model;

        let messages: string | TLLMMessageBlock[] = params?.messages || '';
        let systemInstruction = '';
        const files: BinaryInput[] = params?.files || [];

        // #region Upload files
        const promises = [];
        const _files = [];

        for (let image of files) {
            const binaryInput = BinaryInput.from(image);
            promises.push(binaryInput.upload(AccessCandidate.agent(params.agentId)));

            _files.push(binaryInput);
        }

        await Promise.all(promises);
        // #endregion Upload files

        // If user provide mix of valid and invalid files, we will only process the valid files
        const validFiles = this.getValidFiles(_files, 'all');

        const hasVideo = validFiles.some((file) => file?.mimetype?.includes('video'));

        // GoogleAI only supports one video file at a time
        if (hasVideo && validFiles.length > 1) {
            throw new Error('Only one video file is supported at a time.');
        }

        const fileUploadingTasks = validFiles.map((file) => async () => {
            try {
                const uploadedFile = await this.uploadFile({
                    file,
                    apiKey: (params.credentials as BasicCredentials).apiKey,
                    agentId: params.agentId,
                });

                return { url: uploadedFile.url, mimetype: file.mimetype };
            } catch (error: any) {
                console.error(`Failed to upload file ${file.filename || 'unknown'}:`, error.message);
                return null;
            }
        });

        const uploadedFiles = await processWithConcurrencyLimit(fileUploadingTasks);

        // We throw error when there are no valid uploaded files,
        const validUploadedFiles = uploadedFiles.filter((file) => file !== null);
        if (validUploadedFiles.length === 0) {
            throw new Error(`Failed to upload any files to Google AI Server! All ${uploadedFiles.length} file upload attempts failed.`);
        }

        const fileData = this.getFileData(uploadedFiles);

        const userMessage: TLLMMessageBlock = Array.isArray(messages) ? messages.pop() : { role: TLLMMessageRole.User, content: '' };
        let prompt = userMessage?.content || '';

        // if the the model does not support system instruction, we will add it to the prompt
        if (!MODELS_SUPPORT_SYSTEM_INSTRUCTION.includes(model as string)) {
            prompt = `${prompt}\n${systemInstruction}`;
        }
        //#endregion Separate system message and add JSON response instruction if needed

        // Create proper content structure for files
        const parts = [...fileData, { text: prompt }];

        return [{ parts }];
    }

    private async prepareMessagesWithTools(params: TLLMPreparedParams): Promise<any> {
        let formattedMessages: TLLMMessageBlock[];
        let systemInstruction = '';

        let messages = params?.messages || [];

        const hasSystemMessage = LLMHelper.hasSystemMessage(messages);

        if (hasSystemMessage) {
            const separateMessages = LLMHelper.separateSystemMessages(messages);
            const systemMessageContent = (separateMessages.systemMessage as TLLMMessageBlock)?.content;
            systemInstruction = typeof systemMessageContent === 'string' ? systemMessageContent : '';
            formattedMessages = separateMessages.otherMessages;
        } else {
            formattedMessages = messages;
        }

        // Convert messages to proper Gemini format, ensuring function call/response pairs are maintained
        const processedMessages = this.getConsistentMessages(formattedMessages);

        const toolsPrompt: any = {
            contents: processedMessages as any,
        };

        if (systemInstruction) {
            toolsPrompt.systemInstruction = systemInstruction;
        }

        if (params?.toolsConfig?.tools) toolsPrompt.tools = params?.toolsConfig?.tools as any;
        if (params?.toolsConfig?.tool_choice) {
            toolsPrompt.toolConfig = {
                functionCallingConfig: { mode: (params?.toolsConfig?.tool_choice as any) || 'auto' },
            };
        }

        return toolsPrompt;
    }

    private async prepareMessagesWithTextQuery(params: TLLMPreparedParams): Promise<any[]> {
        const model = params.model;
        let systemInstruction = '';

        const { systemMessage, otherMessages } = LLMHelper.separateSystemMessages(params?.messages as TLLMMessageBlock[]);

        if ('content' in systemMessage) {
            systemInstruction = systemMessage.content as string;
        }

        const responseFormat = params?.responseFormat || '';

        if (responseFormat === 'json') {
            systemInstruction += JSON_RESPONSE_INSTRUCTION;
        }

        // Convert messages to proper Google AI format
        const contents: any[] = [];

        if (otherMessages?.length > 0) {
            for (const message of otherMessages) {
                let textContent = '';

                if (typeof message.content === 'string') {
                    textContent = message.content;
                } else if (Array.isArray(message.content)) {
                    textContent = message.content.map((block: any) => block?.text || '').join(' ');
                } else if (message?.parts?.[0]?.text) {
                    textContent = message.parts[0].text;
                }

                // For models that don't support system instruction, prepend it to first message
                if (!MODELS_SUPPORT_SYSTEM_INSTRUCTION.includes(model as string) && contents.length === 0 && systemInstruction) {
                    textContent = `${systemInstruction}\n\n${textContent}`;
                }

                if (textContent.trim()) {
                    contents.push({
                        parts: [{ text: textContent }],
                    });
                }
            }
        } else {
            // If no messages, create a default empty message
            let textContent = systemInstruction || 'Hello';
            contents.push({
                parts: [{ text: textContent }],
            });
        }

        return contents;
    }

    private async prepareBodyForImageGenRequest(params: TLLMPreparedParams): Promise<any> {
        return {
            prompt: params.prompt,
            model: params.model,
            aspectRatio: (params as any).aspectRatio,
            personGeneration: (params as any).personGeneration,
        };
    }

    // Add this helper method to sanitize function names
    private sanitizeFunctionName(name: string): string {
        // Check if name is undefined or null
        if (name == null) {
            return '_unnamed_function';
        }

        // Remove any characters that are not alphanumeric, underscore, dot, or dash
        let sanitized = name.replace(/[^a-zA-Z0-9_.-]/g, '');

        // Ensure the name starts with a letter or underscore
        if (!/^[a-zA-Z_]/.test(sanitized)) {
            sanitized = '_' + sanitized;
        }

        // If sanitized is empty after removing invalid characters, use a default name
        if (sanitized === '') {
            sanitized = '_unnamed_function';
        }

        // Truncate to 64 characters if longer
        sanitized = sanitized.slice(0, 64);

        return sanitized;
    }

    private async uploadFile({ file, apiKey, agentId }: { file: BinaryInput; apiKey: string; agentId: string }): Promise<{ url: string }> {
        let tempFilePath: string | null = null;

        try {
            if (!apiKey || !file?.mimetype) {
                throw new Error('Missing required parameters to save file for Google AI!');
            }

            // Create a temporary directory
            const tempDir = os.tmpdir();
            const fileName = uid();
            tempFilePath = path.join(tempDir, fileName);

            const bufferData = await file.readData(AccessCandidate.agent(agentId));

            // Write buffer data to temp file
            await fs.promises.writeFile(tempFilePath, new Uint8Array(bufferData));

            // Upload the file using the new GoogleGenAI SDK
            const genAI = new GoogleGenAI({ apiKey });

            // Upload file using the correct Google AI SDK format
            const uploadResponse = await genAI.files.upload({
                file: tempFilePath,
                config: {
                    mimeType: file.mimetype,
                    displayName: fileName,
                },
            });

            // Poll file status until processing is complete
            let uploadedFile = uploadResponse;
            while (uploadedFile.state === 'PROCESSING') {
                process.stdout.write('.');
                // Sleep for 10 seconds
                await new Promise((resolve) => setTimeout(resolve, 10_000));
                // Fetch the file status again
                uploadedFile = await genAI.files.get({ name: uploadedFile.name });
            }

            if (uploadedFile.state === 'FAILED') {
                throw new Error('File processing failed.');
            }

            return {
                url: uploadedFile.uri || uploadedFile.name || '',
            };
        } catch (error) {
            throw new Error(`Error uploading file for Google AI: ${error.message}`);
        } finally {
            // Always clean up temp file, regardless of success or failure
            if (tempFilePath) {
                try {
                    await fs.promises.unlink(tempFilePath);
                } catch (cleanupError) {
                    // Log cleanup failure but don't throw - original error is more important
                    console.warn(`Failed to clean up temporary file ${tempFilePath}:`, cleanupError.message);
                }
            }
        }
    }

    private getValidFiles(files: BinaryInput[], type: 'image' | 'all') {
        const validSources = [];

        for (let file of files) {
            if (this.validMimeTypes[type].includes(file?.mimetype)) {
                validSources.push(file);
            }
        }

        if (validSources?.length === 0) {
            throw new Error(`Unsupported file(s). Please make sure your file is one of the following types: ${this.validMimeTypes[type].join(', ')}`);
        }

        return validSources;
    }

    private getFileData(
        files: {
            url: string;
            mimetype: string;
        }[]
    ): {
        fileData: {
            mimeType: string;
            fileUri: string;
        };
    }[] {
        try {
            const imageData = [];

            for (let file of files) {
                imageData.push({
                    fileData: {
                        mimeType: file.mimetype,
                        fileUri: file.url,
                    },
                });
            }

            return imageData;
        } catch (error) {
            throw error;
        }
    }
}
