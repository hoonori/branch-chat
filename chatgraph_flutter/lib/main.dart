import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';
import 'dart:async';
import 'package:flutter_secure_storage/flutter_secure_storage.dart';

void main() {
  runApp(const ChatGraphApp());
}

class ChatGraphApp extends StatelessWidget {
  const ChatGraphApp({super.key});

  @override
  Widget build(BuildContext context) {
    return const MaterialApp(
      home: ChatScreen(),
      debugShowCheckedModeBanner: false,
    );
  }
}

class ChatScreen extends StatefulWidget {
  const ChatScreen({super.key});

  @override
  State<ChatScreen> createState() => _ChatScreenState();
}

class _ChatScreenState extends State<ChatScreen> {
  final TextEditingController _apiKeyController = TextEditingController();
  final TextEditingController _messageController = TextEditingController();
  final List<Map<String, String>> _messages = [];
  bool _isLoading = false;
  String? _apiKey;
  final _storage = const FlutterSecureStorage();

  @override
  void initState() {
    super.initState();
    _loadApiKey();
  }

  Future<void> _loadApiKey() async {
    final key = await _storage.read(key: 'openai_api_key');
    if (key != null && key.isNotEmpty) {
      setState(() {
        _apiKey = key;
      });
    }
  }

  Future<void> _saveApiKey(String key) async {
    await _storage.write(key: 'openai_api_key', value: key);
    setState(() {
      _apiKey = key;
    });
  }

  Future<void> _sendMessage() async {
    final userMessage = _messageController.text.trim();
    if (userMessage.isEmpty || _apiKey == null || _apiKey!.isEmpty) return;
    setState(() {
      _messages.add({'role': 'user', 'content': userMessage});
      _isLoading = true;
      _messageController.clear();
    });
    try {
      final response = await http
          .post(
            Uri.parse('https://api.openai.com/v1/chat/completions'),
            headers: {
              'Content-Type': 'application/json',
              'Authorization': 'Bearer $_apiKey',
            },
            body: jsonEncode({
              'model': 'gpt-4o',
              'messages': _messages.map((m) => {'role': m['role'], 'content': m['content']}).toList(),
            }),
          )
          .timeout(const Duration(seconds: 20));

      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);
        final reply = data['choices'][0]['message']['content'];
        setState(() {
          _messages.add({'role': 'assistant', 'content': reply});
        });
      } else {
        setState(() {
          _messages.add({'role': 'assistant', 'content': 'Error: ${response.body}'});
        });
      }
    } on TimeoutException {
      setState(() {
        _messages.add({'role': 'assistant', 'content': 'Error: 요청이 너무 오래 걸립니다. 네트워크를 확인하세요.'});
      });
    } catch (e) {
      setState(() {
        _messages.add({'role': 'assistant', 'content': 'Error: $e'});
      });
    } finally {
      setState(() {
        _isLoading = false;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('ChatGraph (Step 1)'), backgroundColor: Colors.blueAccent),
      body: Padding(
        padding: const EdgeInsets.all(12.0),
        child: Column(
          children: [
            if (_apiKey == null)
              Column(
                children: [
                  const Text('Enter your OpenAI API Key:'),
                  TextField(
                    controller: _apiKeyController,
                    decoration: const InputDecoration(hintText: 'sk-...'),
                    obscureText: true,
                  ),
                  const SizedBox(height: 8),
                  ElevatedButton(
                    onPressed: () async {
                      final key = _apiKeyController.text.trim();
                      if (key.isNotEmpty) {
                        await _saveApiKey(key);
                      }
                    },
                    child: const Text('Save API Key'),
                  ),
                  const Divider(),
                ],
              ),
            if (_apiKey != null)
              Expanded(
                child: ListView.builder(
                  itemCount: _messages.length,
                  itemBuilder: (context, idx) {
                    final msg = _messages[idx];
                    final isUser = msg['role'] == 'user';
                    return Align(
                      alignment: isUser ? Alignment.centerRight : Alignment.centerLeft,
                      child: Container(
                        margin: const EdgeInsets.symmetric(vertical: 4),
                        padding: const EdgeInsets.all(10),
                        decoration: BoxDecoration(
                          color: isUser ? Colors.blue[100] : Colors.grey[200],
                          borderRadius: BorderRadius.circular(8),
                        ),
                        child: Text(msg['content'] ?? '', style: const TextStyle(fontSize: 16)),
                      ),
                    );
                  },
                ),
              ),
            if (_apiKey != null)
              Row(
                children: [
                  Expanded(
                    child: TextField(
                      controller: _messageController,
                      onSubmitted: (_) => _sendMessage(),
                      decoration: const InputDecoration(hintText: 'Type your message...'),
                    ),
                  ),
                  IconButton(
                    icon: _isLoading
                        ? const CircularProgressIndicator()
                        : const Icon(Icons.send),
                    onPressed: _isLoading ? null : _sendMessage,
                  ),
                ],
              ),
          ],
        ),
      ),
    );
  }
} 