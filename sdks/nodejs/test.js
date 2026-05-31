const assert = require('assert');
const EvoNetAI = require('./index');

// Mock global fetch for testing
global.fetch = async (url, options) => {
  return {
    ok: true,
    json: async () => ({
      id: "chatcmpl-mock123",
      model: "vietnamese-ai-default",
      choices: [{
        message: { role: "assistant", content: "Xin chào từ EvoNetAI Node SDK!" }
      }]
    })
  };
};

async function runTests() {
  const evonet = new EvoNetAI({ apiKey: "test-key" });
  
  const response = await evonet.chat.completions.create({
    model: "vietnamese-ai-default",
    messages: [{ role: "user", content: "Chào bạn" }]
  });
  
  assert.strictEqual(response.id, "chatcmpl-mock123");
  assert.strictEqual(response.choices[0].message.content, "Xin chào từ EvoNetAI Node SDK!");
  console.log("✅ EvoNetAI Node.js SDK Mock Tests Passed!");
}

runTests().catch(console.error);
