#version 330 core

in vec2 UV;

out vec3 color;

uniform sampler2D textureSampler;

void main(){
    // Output color of the texture at the specified UV coordinate
    color = texture(textureSampler, UV).rgb;
}
