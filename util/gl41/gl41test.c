// Minimal GL 4.1 CORE reproducer for the Luanti bug:
//   geometry renders fine to the default framebuffer, but vanishes when the
//   same geometry is drawn into an FBO whose colour texture is then sampled.
//
// Draws a triangle into an FBO (colour texture + depth texture, exactly like
// Luanti's post-processing target), then samples that texture onto a
// fullscreen quad. Reads back the centre pixel and reports.
//
// Expected if GL is behaving: centre pixel is the triangle's colour.
// If it comes back as the clear colour, the FBO path is dropping geometry and
// we have the bug isolated in ~150 lines instead of a whole engine.
#include <SDL2/SDL.h>
#include <OpenGL/gl3.h>
#include <stdio.h>

static GLuint mkshader(GLenum type, const char *src)
{
	GLuint s = glCreateShader(type);
	glShaderSource(s, 1, &src, NULL);
	glCompileShader(s);
	GLint ok = 0;
	glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
	if (!ok) {
		char log[2048];
		glGetShaderInfoLog(s, sizeof(log), NULL, log);
		printf("SHADER COMPILE FAIL: %s\n", log);
	}
	return s;
}

static GLuint mkprog(const char *vs, const char *fs)
{
	GLuint p = glCreateProgram();
	glAttachShader(p, mkshader(GL_VERTEX_SHADER, vs));
	glAttachShader(p, mkshader(GL_FRAGMENT_SHADER, fs));
	// Luanti binds these explicitly rather than using layout qualifiers,
	// because GLSL 1.50 has no layout(location=) for attributes or outputs.
	glBindAttribLocation(p, 0, "inPos");
	glBindFragDataLocation(p, 0, "outColor");
	glLinkProgram(p);
	GLint ok = 0;
	glGetProgramiv(p, GL_LINK_STATUS, &ok);
	if (!ok) {
		char log[2048];
		glGetProgramInfoLog(p, sizeof(log), NULL, log);
		printf("LINK FAIL: %s\n", log);
	}
	return p;
}

int main(void)
{
	SDL_Init(SDL_INIT_VIDEO);
	SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
	SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 2);
	SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
	SDL_Window *w = SDL_CreateWindow("gl41test", 0, 0, 320, 240,
			SDL_WINDOW_OPENGL | SDL_WINDOW_HIDDEN);
	SDL_GLContext ctx = SDL_GL_CreateContext(w);
	if (!ctx) { printf("no context: %s\n", SDL_GetError()); return 1; }
	printf("GL_VERSION  = %s\n", glGetString(GL_VERSION));
	printf("GLSL        = %s\n", glGetString(GL_SHADING_LANGUAGE_VERSION));

	GLuint vao; glGenVertexArrays(1, &vao); glBindVertexArray(vao);

	// --- FBO with colour texture + DEPTH TEXTURE (as Luanti builds it) ---
	GLuint colorTex, depthTex, fbo;
	glGenTextures(1, &colorTex);
	glBindTexture(GL_TEXTURE_2D, colorTex);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB10_A2, 320, 240, 0, GL_RGBA,
			GL_UNSIGNED_INT_2_10_10_10_REV, NULL);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

	glGenTextures(1, &depthTex);
	glBindTexture(GL_TEXTURE_2D, depthTex);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT24, 320, 240, 0,
			GL_DEPTH_COMPONENT, GL_UNSIGNED_INT, NULL);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

	glGenFramebuffers(1, &fbo);
	glBindFramebuffer(GL_FRAMEBUFFER, fbo);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, colorTex, 0);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, depthTex, 0);
	GLenum st = glCheckFramebufferStatus(GL_FRAMEBUFFER);
	printf("FBO status  = 0x%04x %s\n", st,
			st == GL_FRAMEBUFFER_COMPLETE ? "(COMPLETE)" : "(INCOMPLETE)");

	// --- draw a big triangle into the FBO ---
	const char *vs = "#version 150\nin vec4 inPos;\nvoid main(){ gl_Position = inPos; }\n";
	const char *fs = "#version 150\nout vec4 outColor;\nvoid main(){ outColor = vec4(1.0,0.25,0.0,1.0); }\n";
	GLuint prog = mkprog(vs, fs);

	float tri[] = { -0.9f,-0.9f, 0.9f,-0.9f, 0.0f,0.9f };
	GLuint vbo; glGenBuffers(1, &vbo);
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	glBufferData(GL_ARRAY_BUFFER, sizeof(tri), tri, GL_STATIC_DRAW);

	glViewport(0, 0, 320, 240);
	glClearColor(0.0f, 0.0f, 0.5f, 1.0f);
	glClearDepth(1.0);
	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LEQUAL);
	glDepthMask(GL_TRUE);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	// Luanti draws INDEXED through an element buffer, and its pipeline binds
	// and unbinds the target repeatedly between steps. Mimic both.
	unsigned short idx[3] = {0, 1, 2};
	GLuint ibo; glGenBuffers(1, &ibo);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(idx), idx, GL_STATIC_DRAW);

	glUseProgram(prog);
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, (void *)0);
	for (int pass = 0; pass < 3; pass++) {
		glBindFramebuffer(GL_FRAMEBUFFER, 0);       // detach, like a step switch
		glBindFramebuffer(GL_FRAMEBUFFER, fbo);     // and reattach
		glViewport(0, 0, 320, 240);
		glDrawElements(GL_TRIANGLES, 3, GL_UNSIGNED_SHORT, (void *)0);
	}
	printf("after FBO draw err = 0x%04x\n", glGetError());

	// read the FBO centre pixel directly
	unsigned char px[4] = {0};
	glReadPixels(160, 120, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, px);
	printf("FBO centre pixel   = %d,%d,%d  (expect ~255,64,0 = triangle)\n",
			px[0], px[1], px[2]);

	// --- sample that texture onto the default framebuffer ---
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	glViewport(0, 0, 320, 240);
	glClearColor(0.0f, 0.5f, 0.0f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	const char *qvs = "#version 150\nin vec4 inPos;\nout vec2 uv;\n"
			"void main(){ uv = inPos.xy*0.5+0.5; gl_Position = inPos; }\n";
	const char *qfs = "#version 150\nin vec2 uv;\nuniform sampler2D t;\n"
			"out vec4 outColor;\nvoid main(){ outColor = texture(t, uv); }\n";
	GLuint qprog = mkprog(qvs, qfs);
	float quad[] = { -1,-1,  1,-1,  -1,1,   1,-1,  1,1,  -1,1 };
	GLuint qvbo; glGenBuffers(1, &qvbo);
	glBindBuffer(GL_ARRAY_BUFFER, qvbo);
	glBufferData(GL_ARRAY_BUFFER, sizeof(quad), quad, GL_STATIC_DRAW);
	glUseProgram(qprog);
	glUniform1i(glGetUniformLocation(qprog, "t"), 0);
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, colorTex);
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, (void *)0);
	glDrawArrays(GL_TRIANGLES, 0, 6);
	printf("after quad draw err = 0x%04x\n", glGetError());

	glReadPixels(160, 120, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, px);
	printf("SCREEN centre pixel= %d,%d,%d  (expect ~255,64,0 if sampling works)\n",
			px[0], px[1], px[2]);

	SDL_GL_DeleteContext(ctx);
	SDL_Quit();
	return 0;
}
