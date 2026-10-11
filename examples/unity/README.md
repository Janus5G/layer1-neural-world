# Unity desktop integration example

Requires a desktop Unity project with `System.Net.WebSockets.ClientWebSocket` support (not WebGL). Copy `Assets/Scripts/Layer1BrainClient.cs` into your project, attach it to a movable GameObject, assign a different GameObject as its `target` transform, and run the Layer 1+ bridge separately with `npm run bridge`. In Play Mode, movement comes from the bridge's sampled `north/south/east/west/idle` actions. The example computes bounded distance-improvement rewards. It does **not** supply pathfinding, navigation, guaranteed learning or any complex game mechanics.

No external Unity packages are required by this example, but Unity compatibility **has not been tested in the Unity Editor**. Desktop platforms only; Unity WebGL cannot use this `ClientWebSocket` approach without an alternative transport. A dedicated object may need a Rigidbody or CharacterController integration for production-quality physics; this simple demo changes the Transform directly.

By default `enableLearning` is false for low CPU load. To investigate training, turn it on only in a controlled test with temperature monitoring; a newly created brain begins with random weights. Custom projects must define their own sensor vector, action labels and reward function.
