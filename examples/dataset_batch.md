| KEY | TYPE | SHAPE |
| --- | --- | --- |
| observation.images.cam_high | torch.float32 | torch.Size([1, 3, 480, 640]) |
| observation.images.cam_left_wrist | torch.float32 | torch.Size([1, 3, 480, 640]) |
| observation.images.cam_right_wrist | torch.float32 | torch.Size([1, 3, 480, 640]) |
| observation.state | torch.float32 | torch.Size([1, 14]) |
| observation.effort | torch.float32 | torch.Size([1, 14]) |
| action | torch.float32 | torch.Size([1, 14]) |
| episode_index | torch.int64 | torch.Size([1]) |
| frame_index | torch.int64 | torch.Size([1]) |
| timestamp | torch.float32 | torch.Size([1]) |
| next.done | torch.bool | torch.Size([1]) |
| index | torch.int64 | torch.Size([1]) |
| task_index | torch.int64 | torch.Size([1]) |
| task | 1 | str | ['Open the top cabinet, store the pot inside it then close the cabinet.'] |