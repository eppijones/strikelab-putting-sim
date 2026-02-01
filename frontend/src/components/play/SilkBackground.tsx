import React, { useRef, useMemo, useEffect } from 'react';
import { useFrame, useThree, createPortal } from '@react-three/fiber';
import * as THREE from 'three';

const vertexShader = `
  varying vec2 vUv;
  void main() {
    vUv = uv;
    gl_Position = vec4(position, 1.0); // Render in screen space (clip space)
  }
`;

const fragmentShader = `
  uniform float uTime;
  uniform vec3 uColor;
  varying vec2 vUv;

  // Simplex 2D noise
  vec3 permute(vec3 x) { return mod(((x*34.0)+1.0)*x, 289.0); }

  float snoise(vec2 v){
    const vec4 C = vec4(0.211324865405187, 0.366025403784439,
             -0.577350269189626, 0.024390243902439);
    vec2 i  = floor(v + dot(v, C.yy) );
    vec2 x0 = v -   i + dot(i, C.xx);
    vec2 i1;
    i1 = (x0.x > x0.y) ? vec2(1.0, 0.0) : vec2(0.0, 1.0);
    vec4 x12 = x0.xyxy + C.xxzz;
    x12.xy -= i1;
    i = mod(i, 289.0);
    vec3 p = permute( permute( i.y + vec3(0.0, i1.y, 1.0 ))
    + i.x + vec3(0.0, i1.x, 1.0 ));
    vec3 m = max(0.5 - vec3(dot(x0,x0), dot(x12.xy,x12.xy), dot(x12.zw,x12.zw)), 0.0);
    m = m*m ;
    m = m*m ;
    vec3 x = 2.0 * fract(p * C.www) - 1.0;
    vec3 h = abs(x) - 0.5;
    vec3 ox = floor(x + 0.5);
    vec3 a0 = x - ox;
    m *= 1.79284291400159 - 0.85373472095314 * ( a0*a0 + h*h );
    vec3 g;
    g.x  = a0.x  * x0.x  + h.x  * x0.y;
    g.yz = a0.yz * x12.xz + h.yz * x12.yw;
    return 130.0 * dot(m, g);
  }

  void main() {
    vec2 uv = vUv;
    
    // Create flowing coordinates
    float noiseVal = snoise(uv * 3.0 + uTime * 0.2);
    
    // Distort UVs for the silk effect
    float distortion = snoise(uv * 6.0 - uTime * 0.1) * 0.1;
    
    // Create diagonal waves
    float wave = sin(uv.x * 10.0 + uv.y * 10.0 + uTime + noiseVal * 2.0);
    
    // Soften the wave for a cloth look
    float silk = smoothstep(-1.0, 1.0, wave);
    
    // Add sheen
    float sheen = pow(silk, 2.0) * 0.5;
    
    // Mix base color with darker shadows and lighter highlights
    vec3 finalColor = mix(uColor * 0.5, uColor * 1.5, silk);
    finalColor += sheen * 0.2;
    
    // Add subtle vignette
    float vignette = 1.0 - length(uv - 0.5) * 0.5;
    finalColor *= vignette;

    gl_FragColor = vec4(finalColor, 1.0);
  }
`;

export const SilkBackground: React.FC = () => {
  const meshRef = useRef<THREE.Mesh>(null);
  const scene = useMemo(() => new THREE.Scene(), []);
  const camera = useMemo(() => new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1), []);

  const uniforms = useMemo(
    () => ({
      uTime: { value: 0 },
      uColor: { value: new THREE.Color('#0f172a') }, // Match the slate-900 base
    }),
    []
  );

  useFrame((state, delta) => {
    if (meshRef.current) {
      (meshRef.current.material as THREE.ShaderMaterial).uniforms.uTime.value = state.clock.getElapsedTime();
    }
    
    // Render the background scene first
    state.gl.autoClear = false;
    state.gl.clear();
    state.gl.render(scene, camera);
  }, -1); // Negative priority to run before main scene render

  return createPortal(
    <mesh ref={meshRef}>
      <planeGeometry args={[2, 2]} /> {/* Full screen quad in clip space */}
      <shaderMaterial
        vertexShader={vertexShader}
        fragmentShader={fragmentShader}
        uniforms={uniforms}
        depthWrite={false}
        depthTest={false}
      />
    </mesh>,
    scene
  );
};

