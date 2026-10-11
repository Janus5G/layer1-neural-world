// Unity desktop example (not Unity-WebGL compatible).
// Attach to an agent GameObject and assign target to a scene Transform.
// Physics, rewards and rendering remain entirely in the Unity world.
using System;
using System.Collections.Concurrent;
using System.Net.WebSockets;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using UnityEngine;

public sealed class Layer1BrainClient : MonoBehaviour
{
    [SerializeField] private string serverUrl = "ws://127.0.0.1:9090/engine/ws";
    [SerializeField] private string agentId = "unity-agent-1";
    [SerializeField] private Transform target;
    [SerializeField] private bool enableLearning = false; // CPU-conscious default
    [SerializeField] private float speed = 3f;
    [SerializeField] private float stepSeconds = 0.3f;
    private readonly ConcurrentQueue<string> inbox = new ConcurrentQueue<string>();
    private readonly SemaphoreSlim sendLock = new SemaphoreSlim(1, 1);
    private ClientWebSocket socket;
    private CancellationTokenSource cancellation;
    private bool registered, connected, waiting;
    private float nextStep, nextReward, previousDistance;

    [Serializable]
    private class BridgeCommand
    {
        public string type, agentId;
        public int inputDim;
        public string[] actions;
        public bool learn;
        public float[] observation;
        public float reward;
    }

    [Serializable]
    private class BridgeReply
    {
        public string type, agentId, name, code;
        public int index;
    }

    private async void Start()
    {
        if (target == null) { Debug.LogError("Layer 1+ requires a target Transform."); enabled = false; return; }
        previousDistance = Vector3.Distance(transform.position, target.position);
        cancellation = new CancellationTokenSource();
        socket = new ClientWebSocket();
        try
        {
            await socket.ConnectAsync(new Uri(serverUrl), cancellation.Token);
            connected = true;
            _ = ReceiveLoop(cancellation.Token);
            await Send(new BridgeCommand {
                type = "create", agentId = agentId, inputDim = 6,
                actions = new[] { "north", "south", "east", "west", "idle" }, learn = enableLearning
            });
        }
        catch (Exception e) { Debug.LogWarning("Layer 1+ connection failed: " + e.Message); }
    }

    private void Update()
    {
        while (inbox.TryDequeue(out var raw))
        {
            BridgeReply reply;
            try { reply = JsonUtility.FromJson<BridgeReply>(raw); }
            catch (Exception e) { Debug.LogWarning("Invalid bridge message: " + e.Message); continue; }
            switch (reply.type)
            {
                case "created": registered = true; waiting = false; break;
                case "action":
                    if (reply.agentId != agentId) break;
                    Move(reply.name);
                    float distance = Vector3.Distance(transform.position, target.position);
                    nextReward = Mathf.Clamp(previousDistance - distance, -1f, 1f);
                    previousDistance = distance;
                    waiting = false;
                    break;
                case "error": Debug.LogWarning("Layer 1+ bridge error: " + reply.code); waiting = false; break;
            }
        }
        if (!connected || !registered || waiting || target == null || Time.time < nextStep) return;
        nextStep = Time.time + stepSeconds;
        waiting = true;
        _ = Send(new BridgeCommand {
            type = "step", agentId = agentId, observation = Sense(), reward = nextReward
        });
    }

    // Six bounded observation values in [-1,1], corresponding to the agreed contract.
    private float[] Sense()
    {
        Vector3 p = transform.position, delta = target.position - p;
        return new[] {
            Mathf.Clamp(p.x / 50f, -1, 1), Mathf.Clamp(p.z / 50f, -1, 1),
            Mathf.Clamp(delta.x / 50f, -1, 1), Mathf.Clamp(delta.z / 50f, -1, 1),
            Mathf.Clamp(delta.magnitude / 70f, 0, 1), 1f
        };
    }

    // Move only local Unity objects. The bridge cannot spawn or control game entities.
    private void Move(string action)
    {
        Vector3 direction = action switch {
            "north" => Vector3.forward, "south" => Vector3.back,
            "east" => Vector3.right, "west" => Vector3.left,
            _ => Vector3.zero
        };
        transform.position += direction * (speed * stepSeconds);
    }

    private async Task Send(BridgeCommand message)
    {
        if (socket == null || socket.State != WebSocketState.Open) return;
        await sendLock.WaitAsync();
        try {
            byte[] bytes = Encoding.UTF8.GetBytes(JsonUtility.ToJson(message));
            await socket.SendAsync(new ArraySegment<byte>(bytes), WebSocketMessageType.Text, true, cancellation.Token);
        }
        catch (Exception e) { if (!cancellation.IsCancellationRequested) Debug.LogWarning("Bridge send failed: " + e.Message); }
        finally { sendLock.Release(); }
    }

    private async Task ReceiveLoop(CancellationToken token)
    {
        byte[] chunk = new byte[8192];
        try {
            while (!token.IsCancellationRequested && socket.State == WebSocketState.Open) {
                var buffer = new System.IO.MemoryStream();
                WebSocketReceiveResult packet;
                do {
                    packet = await socket.ReceiveAsync(new ArraySegment<byte>(chunk), token);
                    if (packet.MessageType == WebSocketMessageType.Close) return;
                    buffer.Write(chunk, 0, packet.Count);
                    if (buffer.Length > 16384) return;
                } while (!packet.EndOfMessage);
                inbox.Enqueue(Encoding.UTF8.GetString(buffer.ToArray()));
                buffer.Dispose();
            }
        }
        catch (OperationCanceledException) { }
        catch (Exception e) { if (!token.IsCancellationRequested) inbox.Enqueue("{\"type\":\"error\",\"code\":\"connection_lost\"}"); }
    }

    private void OnDestroy()
    {
        cancellation?.Cancel();
        socket?.Dispose();
        cancellation?.Dispose();
    }
}
