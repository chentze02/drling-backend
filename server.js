import 'dotenv/config';
import express from 'express';
import cors from 'cors';
import bodyParser from 'body-parser';
import { GoogleGenAI } from '@google/genai';

const app = express();
const port = process.env.PORT || 5001;

// --- Middleware ---
app.use(cors());
app.use(bodyParser.json({ limit: '10mb' }));

// --- Initialize Google GenAI Client ---
const API_KEY = process.env.GOOGLE_API_KEY;
if (!API_KEY) {
  throw new Error("GOOGLE_API_KEY is not defined in the .env file.");
}
const ai = new GoogleGenAI(API_KEY);

// Health Check Endpoint
app.get('/', (req, res) => {
  res.status(200).send('Smile API server is running!');
});


// --- Main API Endpoint ---
app.post('api/analyze-smile', async (req, res) => {
  const { imageData } = req.body;
  if (!imageData) {
    return res.status(400).json({ error: 'imageData is required.' });
  }

  try {
    const base64Data = imageData.split(',')[1];
    const mimeType = imageData.substring(imageData.indexOf(':') + 1, imageData.indexOf(';'));
    
    const imagePart = {
      inlineData: { mimeType, data: base64Data },
    };

    const imagePrompt = "Create a realistic dental transformation of this smile showing: perfectly straight and aligned teeth, professionally whitened teeth (bright white but natural-looking), no gaps, no cavities, a symmetrical smile, and healthy pink gums. Keep the person's natural facial structure identical. Make it look like a professional, realistic cosmetic dentistry result.";
    const analysisPrompt = `Analyze this dental smile photo and provide a concise treatment plan in this EXACT format:\n\n**Current Concerns:**\n• [List 2-3 main issues]\n\n**Recommended Treatments:**\n• [Treatment 1] - [One sentence description]\n• [Treatment 2] - [One sentence description]\n\n**Timeline:**\n• Estimated duration: [X-X months]\n\n**Expected Results:**\n• [2-3 sentences describing the final outcome]`;

    // --- Run API calls (no changes here) ---
    const [imageResult, analysisResult] = await Promise.all([
      ai.models.generateContent({
        model: 'gemini-2.5-flash-image',
        contents: [{ role: 'user', parts: [{text: imagePrompt}, imagePart] }],
      }),
      ai.models.generateContent({
        model: 'gemini-2.5-flash-image',
        contents: [{ role: 'user', parts: [{text: analysisPrompt}, imagePart] }],
      }),
    ]);

    // --- NEW: Robustly process the responses ---

    // Safely find the image part from the response
    const imageResponsePart = imageResult.candidates?.[0]?.content?.parts.find(part => part.inlineData);
    const transformedImageData = imageResponsePart?.inlineData?.data;

    // Safely find the text part from the response
    const textResponsePart = analysisResult.candidates?.[0]?.content?.parts.find(part => part.text);
    const recommendationsText = textResponsePart?.text;

    // Add a final check to ensure we got both parts
    if (!transformedImageData || !recommendationsText) {
      console.error("Incomplete AI Response:", { imageResult, analysisResult });
      throw new Error("The AI model did not return the expected content. This could be due to safety filters. Please try a different photo.");
    }
    
    // --- Send successful response (no changes here) ---
    res.status(200).json({
      transformedImage: `data:${mimeType};base64,${transformedImageData}`,
      recommendations: recommendationsText
    });

  } catch (error) {
    console.error('Error in /api/analyze-smile:', error);
    res.status(500).json({ error: 'Failed to analyze smile.', details: error.message });
  }
});

// --- Start Server ---
app.listen(port, () => {
  console.log(`Smile analyzer backend listening at http://localhost:${port}`);
});