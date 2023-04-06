#ifndef TINYPBS
#define TINYPBS

/*
				      __
			     /\    .-" /
			    /  ; .'  .' 
			   :   :/  .'   
			    \  ;-.'     
	       .--""""--..__/     `.    
	     .'           .'    `o  \   
	    /                    `   ;  
	   :                  \      :  
	 .-;        -.         `.__.-'  
	:  ;          \     ,   ;       
	'._:           ;   :   (        
	    \/  .__    ;    \   `-.     
	      ;     "-,/_..--"`-..__)    
	     '""--.._:

	A tiny implementation of physically based shading model for S.T.A.L.K.E.R. Clear Sky.

	Features:
	- Metallic - Roughness workflow
	- Filtered Importance Sampling
	- GGX + Lambert BRDFs

	Author:
	- LVutner

	Credits:
	- Michal Iwanicki & Angelo Pesce (DFG approximation)
	- xroo (Test map where I've been testing this shader)
	- Caesar (CS port of test map from xroo)
	- Blazej Kozlowski (Bunny ASCII art)

	References:
	- https://developer.nvidia.com/gpugems/gpugems3/part-iii-rendering/chapter-20-gpu-based-importance-sampling
	- http://miciwan.com/SIGGRAPH2015/course_notes_wip.pdf
	- https://graphicrants.blogspot.com/2013/08/specular-brdf-reference.html
	- https://schuttejoe.github.io/post/ggximportancesamplingpart1/
	- https://www.unrealengine.com/en-US/blog/physically-based-shading-on-mobile
*/

//The settings
#define LV_REMAP_LIGHT_DIRECTION //Remaps light direction for GSC cubemaps (Enable if you're using vanilla engine)
#define LV_ENABLE_FIS //Enables filtered importance sampling for specular (Disable if you prefiltered cubemaps offline)
#define LV_ENABLE_BRIGHTNESS_HACK //Enables brightness hack (Multiply IBL by s_tonemap)
#define LV_FIS_SAMPLE_COUNT 32 //Quality of FIS (filtered importance sampling)
#define LV_FIS_MIP_BIAS 1.0 //Mip bias for FIS

//Hardcoded constants
#define LV_STATIC_METALNESS 0.0 //Metalness value for "flat" geometry
#define LV_STATIC_ROUGHNESS 0.5 //Roughness value for "flat" geometry

//UDN normal blending
float3 blend_normal(float3 normal_0, float3 normal_1)
{
	return normalize(float3(normal_0.xy + normal_1.xy, normal_0.z));
}

//Normal reconstruction
float3 reconstruct_normal(float2 normal)
{
	float3 normal_unpacked;
	normal_unpacked.xy = normal * 2.0 - 1.0;
	normal_unpacked.z = sqrt(1.0 - saturate(dot(normal_unpacked.xy, normal_unpacked.xy)));
	return normal_unpacked;
}

//Samples two cubemaps and interpolates them
float3 sample_sky(float3 direction, float mip_level, float blend_factor, TextureCube t_current, TextureCube t_next)
{
	float3 cubemap_sample = t_current.SampleLevel(smp_rtlinear, direction, mip_level).xyz;
	return lerp(cubemap_sample, t_next.SampleLevel(smp_rtlinear, direction, mip_level).xyz, blend_factor);
}

//Calculates mip level for FIS
//https://developer.nvidia.com/gpugems/gpugems3/part-iii-rendering/chapter-20-gpu-based-importance-sampling
float fis_calculate_mip(int sample_count, float pdf, float cubemap_width)
{
    float sa_sample = 1.0 / (float(sample_count) * pdf); //Solid angle of the sample
    float sa_pixel = 12.56 / (6.0 * cubemap_width * cubemap_width); //Solid angle of the pixel
    return max(0.5 * log2(sa_sample / sa_pixel) + LV_FIS_MIP_BIAS, 0.0);
}

//Cheap approximation of DFG term
//http://miciwan.com/SIGGRAPH2015/course_notes_wip.pdf
float2 ggx_dfg(in float NdotV, in float roughness)
{
    float alpha = roughness * roughness;
	float bias = pow(2.0, -(7.0 * NdotV + 4.0 * alpha));
	float scale = 1.0 - bias - alpha * max(bias, min(roughness, 0.739 + 0.323 * NdotV) - 0.434);
    return float2(scale, bias);	
}

//Normal distribution function GGX
//https://graphicrants.blogspot.com/2013/08/specular-brdf-reference.html
float ggx_ndf(float NdotH, float alpha)
{
    alpha *= alpha; //Remap
    float denominator = (NdotH * alpha - NdotH) * NdotH + 1.0;
    return alpha / (3.14 * denominator * denominator);
}

//Samples microfacet normals of GGX distribution
//https://schuttejoe.github.io/post/ggximportancesamplingpart1/
float3 sample_ggx_ndf(float2 rng, float alpha)
{
	float phi = 6.28 * rng.x;

	float cos_theta = sqrt((1.0 - rng.y) / (1.0 + (alpha * alpha - 1.0) * rng.y));
	float sin_theta = sqrt(1.0 - cos_theta * cos_theta);

	float3 H_tangent;
	sincos(phi, H_tangent.y, H_tangent.x);
	H_tangent.xy *= sin_theta;
	H_tangent.z = cos_theta;

	return H_tangent; 
}

//Calculates direct shading components
float4 get_direct_shading(float3 N, float3 L, float3 H, float roughness)
{
    float alpha = roughness * roughness * roughness * roughness; //Disney remap
    float2 NdotL_H = saturate(float2(dot(N, L), dot(N, H)));

    float GGX = (NdotL_H.y * alpha - NdotL_H.y) * NdotL_H.y + 1.0;
    GGX = alpha / (GGX * GGX); //No PI in denominator, we gonna apply it later
    return float4(NdotL_H.xxx / 3.14, clamp(GGX, 0.0, 32.0)); //Avoid overflow. We do not have HDR, nor we can allow GGX to go insane...
}

//Filtered importance sampling
//https://developer.nvidia.com/gpugems/gpugems3/part-iii-rendering/chapter-20-gpu-based-importance-sampling
float3 fis_cubemap(float3 R, float alpha, float cubemap_width, float blend_factor, TextureCube t_current, TextureCube t_next)
{
	//Handle mirror reflections
	if(alpha <= 0.0)
		return sample_sky(R, 0.0, blend_factor, t_current, t_next);

	//Isotropic assumption
	float3 N = R;
    float3 V = R;

	//Build orthonormal basis
	//This was taken from UE4 paper.
	float3 U = abs(N.z) < 0.999 ? float3(0.0, 0.0, 1.0) : float3(1.0, 0.0, 0.0);
	float3 T = normalize(cross(U, N));
	float3 B = cross(N, T);
	
	//Rotate view direction to tangent space
	float3 V_tangent = normalize(mul(V, transpose(float3x3(T, B, N))));

	//Accumulator
	float4 lighting_weight = float4(0.0, 0.0, 0.0, 0.0);

	for(int i = 0; i < LV_FIS_SAMPLE_COUNT; i++)
	{
		//https://www.shadertoy.com/view/mts3zN
		float2 rng = frac(0.5 + float(i) * float2(0.245122333753, 0.430159709002));

		//Microfacet normal in tangent space
		float3 H_tangent = sample_ggx_ndf(rng, alpha);

		//Light direction in tangent space
		float3 L_tangent = reflect(-V_tangent, H_tangent);

		//NdotL and NdotH. Clamped between 1e-5 - 1.0 due to NaNs...
		float2 NdotL_H = clamp(float2(L_tangent.z, H_tangent.z), 1e-5, 1.0);

		//Calculate PDF
		//D * NdotH / (4.0 * VdotH) =>
		//D * VdotH / (4.0 * VdotH) =>
		//D / 4.0
		//This simplification comes from isotropic assumption.
		float pdf = ggx_ndf(NdotL_H.y, alpha) * 0.25;

		//Calculate mip level
		float mip_level = fis_calculate_mip(LV_FIS_SAMPLE_COUNT, pdf, cubemap_width);

		//Rotate tangent vector to world space
		float3 L = (T * L_tangent.x) + (B * L_tangent.y) + (N * L_tangent.z);

		#ifdef LV_REMAP_LIGHT_DIRECTION
			//Remap the light direction. It's required for vanilla cubemaps.
			float3 vabs = abs(L);
			L.xyz /= max(vabs.x, max(vabs.y, vabs.z));
			L.y = L.y < 0.999 ? L.y * 2.0 - 1.0 : L.y;
		#endif

		//Sample cubemap with new direction and mip level
		float3 sampled_cube = sample_sky(L, mip_level, blend_factor, t_current, t_next).xyz;

		//Weight the sample by NdotL and accumulate
		lighting_weight.xyz += sampled_cube * NdotL_H.x;
		lighting_weight.w += NdotL_H.x;
	}

	//Output
	return lighting_weight.xyz * (1.0 / lighting_weight.w);
}

//Whole magic starts here...
//Inspired by: https://www.unrealengine.com/en-US/blog/physically-based-shading-on-mobile
float3 get_shading(
	float3 vP, float3 vN, //View position and normal
	float3x4 view_to_world, //Inverted view matrix
	float4 light_buffer, //Light buffer (s_accumulator)
	float3 albedo, //Albedo
	float metalness, float roughness, float ambient_occlusion, //Self-explainatory
	float exposure, float blend_factor, //s_tonemap - exposure, and blend factor (env_color.w)
	TextureCube t_current, TextureCube t_next) //Cubemap textures
{
	//Diffuse and specular color for metal-rough workflow
	float3 diffuse_color = (1.0 - metalness) * albedo;
	float3 specular_color = lerp((float3)0.04, albedo, metalness);

	//GGX alpha
	float alpha = roughness * roughness;
	
	//World space vectors
	float3 P = mul(view_to_world, float4(vP.xyz, 1.0)).xyz; //Position
	float3 N = mul(view_to_world, float4(vN.xyz, 0.0)).xyz; //Normal
	float3 V = -normalize(P - eye_position); //View direction
	float3 R = reflect(-V, N); //Reflection direction

	//DFG LUT
	float2 brdf_lut = ggx_dfg(saturate(dot(N, V)), roughness);
	float3 dfg_scaled = specular_color * brdf_lut.x + brdf_lut.y;

	//Get dimensions of current sky texture. We only need width of mip0
	uint width, height, levels;
	t_current.GetDimensions(0, width, height, levels);	

	//Diffuse IBL (Approximated by setting highest mip possible during cube sampling)
	//Pi or not to Pi... Up to you, you can divide it by PI.
	float3 ibl_diffuse = sample_sky(N, 10.0, blend_factor, t_current, t_next);

	//Specular IBL (Approximated with filtered importance sampling)
#ifdef LV_ENABLE_FIS
	float3 ibl_specular = fis_cubemap(R, alpha, width, blend_factor, t_current, t_next);
#else
	float3 ibl_specular = sample_sky(R, float(levels) * roughness, blend_factor, t_current, t_next);
#endif

	//Maintain somewhat OK brightness...
#ifdef LV_ENABLE_BRIGHTNESS_HACK
	ibl_diffuse *= exposure;
	ibl_specular *= exposure;
#endif

	//Combine directional shading - the tricky part.
	//(diffusecolor) * (NdotL/PI * LightColor * Shadow);
	//(NDF_without_PI * FG) * (NdotL/PI * LightColor * Shadow);
	//And we end up with somewhat physically plausible GGX+Lambert
	float3 direct = (light_buffer.www * dfg_scaled + diffuse_color) * light_buffer.xyz;
	
	//Combine indirect lighting
	float3 indirect = (ibl_specular * dfg_scaled) + (ibl_diffuse * (1.0 - dfg_scaled) * diffuse_color);

	//Apply ambient occlusion, output
	return indirect * ambient_occlusion + direct;
}
#endif
