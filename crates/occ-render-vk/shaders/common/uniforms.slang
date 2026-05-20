// Push constants + BDA helpers. Pointer fields use Slang's T* syntax.

import types;

struct PushConstants {
    FrameData* frame_data;
    uint object_index;
    uint _pad;
};

[vk::push_constant]
ConstantBuffer<PushConstants> g_push;
