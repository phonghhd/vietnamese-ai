/**
 * EvoNetAI SDK for Node.js
 * Compatible with OpenAI SDK syntax.
 */

class EvoNetAI {
  constructor(config = {}) {
    this.apiKey = config.apiKey || process.env.EVONET_API_KEY || 'default-key';
    this.baseURL = config.baseURL || 'http://localhost:8000/v1';
    
    this.chat = {
      completions: {
        create: async (params) => this._createChatCompletion(params)
      }
    };
  }

  async _createChatCompletion(params) {
    const url = `${this.baseURL}/chat/completions`;
    const headers = {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${this.apiKey}`
    };

    try {
      const response = await fetch(url, {
        method: 'POST',
        headers: headers,
        body: JSON.stringify(params)
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(`EvoNet API Error: ${response.status} - ${JSON.stringify(errorData)}`);
      }

      return await response.json();
    } catch (error) {
      throw new Error(`Failed to call EvoNetAI: ${error.message}`);
    }
  }
}

module.exports = EvoNetAI;
