<?php

namespace EvoNetAI;

class ChatCompletions {
    private $client;

    public function __construct($client) {
        $this->client = $client;
    }

    public function create(array $params) {
        return $this->client->request('POST', '/chat/completions', $params);
    }
}

class EvoNetAI {
    private $apiKey;
    private $baseUrl;
    public $chat;

    public function __construct(string $apiKey = '', string $baseUrl = 'http://localhost:8000/v1') {
        $this->apiKey = $apiKey ?: getenv('EVONET_API_KEY');
        $this->baseUrl = rtrim($baseUrl, '/');
        
        $this->chat = new \stdClass();
        $this->chat->completions = new ChatCompletions($this);
    }

    public function request(string $method, string $endpoint, array $data = []) {
        $url = $this->baseUrl . $endpoint;
        
        $ch = curl_init($url);
        
        $headers = [
            'Content-Type: application/json',
            'Authorization: Bearer ' . $this->apiKey
        ];
        
        curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);
        curl_setopt($ch, CURLOPT_HTTPHEADER, $headers);
        
        if ($method === 'POST') {
            curl_setopt($ch, CURLOPT_POST, true);
            curl_setopt($ch, CURLOPT_POSTFIELDS, json_encode($data));
        }
        
        $response = curl_exec($ch);
        $httpCode = curl_getinfo($ch, CURLINFO_HTTP_CODE);
        
        if (curl_errno($ch)) {
            throw new \Exception('EvoNet API Curl Error: ' . curl_error($ch));
        }
        
        curl_close($ch);
        
        $decoded = json_decode($response, true);
        
        if ($httpCode >= 400) {
            throw new \Exception('EvoNet API Error: HTTP ' . $httpCode . ' - ' . $response);
        }
        
        return $decoded;
    }
}
