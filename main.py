"""
Modelador 3D - Trabalho Final de Computação Gráfica
UNIOESTE - 2025
Pipeline de Alvy Ray Smith
"""

import pygame
import numpy as np
import sys

# ============================================================================
# TRANSFORMAÇÕES GEOMÉTRICAS
# ============================================================================

def translation_matrix(t):
    """Matriz de translação 4x4"""
    T = np.eye(4, dtype=np.float64)
    T[0, 3], T[1, 3], T[2, 3] = t[0], t[1], t[2]
    return T

def rotation_matrix(angle, axis):
    """Matriz de rotação 4x4 usando fórmula de Rodrigues"""
    axis = axis / np.linalg.norm(axis)
    theta = np.radians(angle)
    c, s, t = np.cos(theta), np.sin(theta), 1 - np.cos(theta)
    x, y, z = axis
    
    R = np.eye(4, dtype=np.float64)
    R[:3, :3] = [
        [t*x*x + c,    t*x*y - s*z,  t*x*z + s*y],
        [t*x*y + s*z,  t*y*y + c,    t*y*z - s*x],
        [t*x*z - s*y,  t*y*z + s*x,  t*z*z + c]
    ]
    return R

def scaling_matrix(s):
    """Matriz de escala 4x4"""
    S = np.eye(4, dtype=np.float64)
    S[0, 0] = S[1, 1] = S[2, 2] = s
    return S

# ============================================================================
# CUBO
# ============================================================================

class Cube:
    """Cubo 3D com transformações geométricas"""
    
    def __init__(self, position=None):
        if position is None:
            position = np.array([0.0, 0.0, 0.0])
        self.position = position
        
        # Vértices locais do cubo unitário
        s = 0.5
        self.local_vertices = np.array([
            [-s, -s, -s, 1], [ s, -s, -s, 1], [ s,  s, -s, 1], [-s,  s, -s, 1],
            [-s, -s,  s, 1], [ s, -s,  s, 1], [ s,  s,  s, 1], [-s,  s,  s, 1]
        ], dtype=np.float64)
        
        # Faces (índices dos vértices)
        self.faces = [
            [0, 1, 2, 3], [4, 7, 6, 5], [0, 4, 5, 1],
            [2, 6, 7, 3], [0, 3, 7, 4], [1, 5, 6, 2]
        ]
        
        # Normais das faces
        self.face_normals = np.array([
            [0, 0, -1], [0, 0, 1], [0, -1, 0],
            [0, 1, 0], [-1, 0, 0], [1, 0, 0]
        ], dtype=np.float64)
        
        # Matriz de transformação
        self.transform_matrix = translation_matrix(position)
        
        # Material
        self.color = np.array([0.7, 0.7, 0.7])
        self.ambient = 0.2
        self.diffuse = 0.7
        self.specular = 0.5
        self.shininess = 32.0
        
    def translate(self, delta):
        self.position += delta
        self.transform_matrix = translation_matrix(delta) @ self.transform_matrix
        
    def rotate(self, angle, axis):
        T1 = translation_matrix(-self.position)
        R = rotation_matrix(angle, axis)
        T2 = translation_matrix(self.position)
        self.transform_matrix = T2 @ R @ T1 @ self.transform_matrix
        
    def scale(self, factor):
        T1 = translation_matrix(-self.position)
        S = scaling_matrix(factor)
        T2 = translation_matrix(self.position)
        self.transform_matrix = T2 @ S @ T1 @ self.transform_matrix
        
    def get_transformed_vertices(self):
        return (self.transform_matrix @ self.local_vertices.T).T
        
    def get_transformed_normals(self):
        M = self.transform_matrix[:3, :3]
        normals = (M @ self.face_normals.T).T
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return normals / norms
        
    def set_color(self, color):
        self.color = np.array(color)

# ============================================================================
# CÂMERA
# ============================================================================

class Camera:
    """Câmera virtual com projeção perspectiva"""
    
    def __init__(self, width, height):
        self.position = np.array([0.0, 2.0, 8.0])
        self.target = np.array([0.0, 0.0, 0.0])
        self.up = np.array([0.0, 1.0, 0.0])
        self.fov = 60.0
        self.aspect_ratio = width / height
        self.near = 0.1
        self.far = 100.0
        self.width = width
        self.height = height
        self.orbit_theta = 0.0
        self.orbit_phi = 20.0
        self.orbit_distance = np.linalg.norm(self.position - self.target)
        
    def get_view_matrix(self):
        """Matriz de visão (world -> camera)"""
        z = self.position - self.target
        z = z / np.linalg.norm(z)
        x = np.cross(self.up, z)
        x = x / np.linalg.norm(x)
        y = np.cross(z, x)
        
        view = np.eye(4, dtype=np.float64)
        view[0, :3], view[1, :3], view[2, :3] = x, y, z
        view[0, 3] = -np.dot(x, self.position)
        view[1, 3] = -np.dot(y, self.position)
        view[2, 3] = -np.dot(z, self.position)
        return view
        
    def get_projection_matrix(self):
        """Matriz de projeção perspectiva"""
        f = 1.0 / np.tan(np.radians(self.fov) / 2.0)
        proj = np.zeros((4, 4))
        proj[0, 0] = f / self.aspect_ratio
        proj[1, 1] = f
        proj[2, 2] = (self.far + self.near) / (self.near - self.far)
        proj[2, 3] = (2.0 * self.far * self.near) / (self.near - self.far)
        proj[3, 2] = -1.0
        return proj
        
    def rotate_around_target(self, delta_theta, delta_phi):
        """Rotação orbital da câmera"""
        self.orbit_theta += delta_theta
        self.orbit_phi = np.clip(self.orbit_phi + delta_phi, -89.0, 89.0)
        
        theta = np.radians(self.orbit_theta)
        phi = np.radians(self.orbit_phi)
        
        x = self.orbit_distance * np.cos(phi) * np.sin(theta)
        y = self.orbit_distance * np.sin(phi)
        z = self.orbit_distance * np.cos(phi) * np.cos(theta)
        
        self.position = self.target + np.array([x, y, z])

# ============================================================================
# LUZ
# ============================================================================

class Light:
    """Fonte de luz pontual"""
    
    def __init__(self):
        self.position = np.array([5.0, 8.0, 10.0])
        self.ambient = np.array([0.3, 0.3, 0.3])
        self.diffuse = np.array([1.0, 1.0, 1.0])
        self.specular = np.array([1.0, 1.0, 1.0])

# ============================================================================
# RENDERER (Pipeline de Alvy Ray Smith + Z-buffer)
# ============================================================================

class Renderer:
    """Renderizador com Pipeline de Alvy Ray Smith"""
    
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.z_buffer = np.full((height, width), float('inf'))
        self.frame_buffer = np.zeros((height, width, 3), dtype=np.uint8)
        
    def clear_buffers(self):
        self.z_buffer.fill(float('inf'))
        self.frame_buffer.fill(0)
        
    def render_scene(self, screen, objects, camera, light, shading_mode, selected):
        self.clear_buffers()
        view_matrix = camera.get_view_matrix()
        proj_matrix = camera.get_projection_matrix()
        
        for obj in objects:
            self.render_object(obj, view_matrix, proj_matrix, camera, light, 
                             shading_mode, obj == selected)
        
        surf = pygame.surfarray.make_surface(np.transpose(self.frame_buffer, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        
    def render_object(self, obj, view_matrix, proj_matrix, camera, light, 
                     shading_mode, is_selected):
        # Pipeline: Model -> View -> Projection -> NDC -> Screen
        vertices_world = obj.get_transformed_vertices()
        vertices_view = (view_matrix @ vertices_world.T).T
        vertices_clip = (proj_matrix @ vertices_view.T).T
        
        # Divisão perspectiva
        vertices_ndc = np.zeros_like(vertices_clip)
        for i in range(len(vertices_clip)):
            w = vertices_clip[i, 3]
            vertices_ndc[i] = vertices_clip[i] / w if abs(w) > 1e-10 else vertices_clip[i]
        
        # Viewport transform
        vertices_screen = np.zeros((len(vertices_ndc), 3))
        for i in range(len(vertices_ndc)):
            vertices_screen[i, 0] = (vertices_ndc[i, 0] + 1.0) * 0.5 * self.width
            vertices_screen[i, 1] = (1.0 - vertices_ndc[i, 1]) * 0.5 * self.height
            vertices_screen[i, 2] = vertices_ndc[i, 2]
        
        normals = obj.get_transformed_normals()
        
        # Rasterizar cada face
        for face_idx, face in enumerate(obj.faces):
            face_verts_screen = vertices_screen[face]
            face_verts_world = vertices_world[face]
            
            # Back-face culling
            if not self.is_front_facing(face_verts_screen):
                continue
            
            normal = normals[face_idx]
            
            # Sombreamento
            if shading_mode == 'constant':
                color = self.constant_shading(face_verts_world, normal, obj, light, camera)
            else:
                color = self.phong_shading(face_verts_world, normal, obj, light, camera)
            
            if is_selected:
                color = color * 0.7 + np.array([0.3, 0.3, 0.0])
            
            self.rasterize_polygon(face_verts_screen, color)
    
    def is_front_facing(self, verts):
        v0, v1, v2 = verts[0][:2], verts[1][:2], verts[2][:2]
        edge1, edge2 = v1 - v0, v2 - v0
        return (edge1[0] * edge2[1] - edge1[1] * edge2[0]) > 0
    
    def constant_shading(self, verts, normal, obj, light, camera):
        """Sombreamento constante (flat)"""
        center = np.mean(verts[:, :3], axis=0)
        L = light.position - center
        L = L / np.linalg.norm(L)
        V = camera.position - center
        V = V / np.linalg.norm(V)
        
        ambient = obj.ambient * light.ambient * obj.color
        diffuse = obj.diffuse * max(0, np.dot(normal, L)) * light.diffuse * obj.color
        
        R = 2.0 * np.dot(normal, L) * normal - L
        R = R / np.linalg.norm(R)
        specular = obj.specular * (max(0, np.dot(R, V)) ** obj.shininess) * light.specular
        
        return np.clip(ambient + diffuse + specular, 0, 1)
    
    def phong_shading(self, verts, normal, obj, light, camera):
        """Sombreamento Phong (simplificado)"""
        center = np.mean(verts[:, :3], axis=0)
        L = light.position - center
        L = L / np.linalg.norm(L)
        V = camera.position - center
        V = V / np.linalg.norm(V)
        
        ambient = obj.ambient * 0.8 * light.ambient * obj.color
        diffuse = obj.diffuse * 1.2 * max(0, np.dot(normal, L)) * light.diffuse * obj.color
        
        H = L + V
        H = H / np.linalg.norm(H)
        specular = obj.specular * 1.5 * (max(0, np.dot(normal, H)) ** (obj.shininess * 1.5)) * light.specular
        
        return np.clip(ambient + diffuse + specular, 0, 1)
    
    def rasterize_polygon(self, verts, color):
        """Rasterização com Z-buffer"""
        color_rgb = (color * 255).astype(np.uint8)
        
        xs, ys = verts[:, 0], verts[:, 1]
        x_min = int(max(0, np.floor(xs.min())))
        x_max = int(min(self.width - 1, np.ceil(xs.max())))
        y_min = int(max(0, np.floor(ys.min())))
        y_max = int(min(self.height - 1, np.ceil(ys.max())))
        
        for y in range(y_min, y_max + 1):
            for x in range(x_min, x_max + 1):
                if self.point_in_polygon(x, y, verts[:, :2]):
                    z = self.interpolate_z(x, y, verts)
                    if z < self.z_buffer[y, x]:
                        self.z_buffer[y, x] = z
                        self.frame_buffer[y, x] = color_rgb
    
    def point_in_polygon(self, x, y, verts):
        """Teste de ponto em polígono (ray casting)"""
        n, inside = len(verts), False
        p1x, p1y = verts[0]
        for i in range(1, n + 1):
            p2x, p2y = verts[i % n]
            if y > min(p1y, p2y) and y <= max(p1y, p2y) and x <= max(p1x, p2x):
                if p1y != p2y:
                    xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                if p1x == p2x or x <= xinters:
                    inside = not inside
            p1x, p1y = p2x, p2y
        return inside
    
    def interpolate_z(self, x, y, verts):
        """Interpolação de profundidade (coordenadas baricêntricas)"""
        v0, v1, v2 = verts[0], verts[1], verts[2]
        denom = (v1[1] - v2[1]) * (v0[0] - v2[0]) + (v2[0] - v1[0]) * (v0[1] - v2[1])
        if abs(denom) < 1e-10:
            return v0[2]
        w0 = ((v1[1] - v2[1]) * (x - v2[0]) + (v2[0] - v1[0]) * (y - v2[1])) / denom
        w1 = ((v2[1] - v0[1]) * (x - v2[0]) + (v0[0] - v2[0]) * (y - v2[1])) / denom
        w2 = 1.0 - w0 - w1
        return w0 * v0[2] + w1 * v1[2] + w2 * v2[2]

# ============================================================================
# APLICAÇÃO PRINCIPAL
# ============================================================================

class Modeler3D:
    def __init__(self):
        pygame.init()
        self.width, self.height = 1400, 900
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Modelador 3D - Pipeline de Alvy Ray Smith")
        
        self.objects = []
        self.camera = Camera(self.width, self.height)
        self.light = Light()
        self.renderer = Renderer(self.width, self.height)
        
        self.shading_mode = 'phong'
        self.selected = None
        self.mouse_down = False
        self.last_mouse_pos = None
        self.clock = pygame.time.Clock()
        self.running = True
        
        # Fonte
        pygame.font.init()
        self.font = pygame.font.Font(None, 20)
        
        # Adicionar cubos iniciais
        cube1 = Cube(np.array([-2.0, 0.0, -5.0]))
        cube1.set_color([0.8, 0.2, 0.2])
        cube2 = Cube(np.array([2.0, 0.0, -5.0]))
        cube2.set_color([0.2, 0.2, 0.8])
        self.objects = [cube1, cube2]
        self.selected = cube1
        
    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                self.handle_keyboard(event)
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                self.mouse_down = True
                self.last_mouse_pos = pygame.mouse.get_pos()
            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                self.mouse_down = False
            elif event.type == pygame.MOUSEMOTION and self.mouse_down:
                dx = event.pos[0] - self.last_mouse_pos[0]
                dy = event.pos[1] - self.last_mouse_pos[1]
                self.camera.rotate_around_target(dx * 0.3, dy * 0.3)
                self.last_mouse_pos = event.pos
                
    def handle_keyboard(self, event):
        if event.key == pygame.K_F1:
            self.shading_mode = 'phong' if self.shading_mode == 'constant' else 'constant'
        elif event.key == pygame.K_n:
            cube = Cube(np.array([0, 0, -5]))
            cube.set_color([np.random.rand(), np.random.rand(), np.random.rand()])
            self.objects.append(cube)
            self.selected = cube
        elif event.key == pygame.K_TAB:
            if self.objects:
                idx = (self.objects.index(self.selected) + 1) % len(self.objects) if self.selected in self.objects else 0
                self.selected = self.objects[idx]
        elif event.key == pygame.K_DELETE and self.selected in self.objects:
            self.objects.remove(self.selected)
            self.selected = None
        elif event.key == pygame.K_ESCAPE:
            self.running = False
            
        if self.selected:
            if event.key == pygame.K_w:
                self.selected.translate(np.array([0, 0.3, 0]))
            elif event.key == pygame.K_s:
                self.selected.translate(np.array([0, -0.3, 0]))
            elif event.key == pygame.K_a:
                self.selected.translate(np.array([-0.3, 0, 0]))
            elif event.key == pygame.K_d:
                self.selected.translate(np.array([0.3, 0, 0]))
            elif event.key == pygame.K_q:
                self.selected.translate(np.array([0, 0, 0.3]))
            elif event.key == pygame.K_e:
                self.selected.translate(np.array([0, 0, -0.3]))
            elif event.key == pygame.K_x:
                self.selected.rotate(15, np.array([1, 0, 0]))
            elif event.key == pygame.K_y:
                self.selected.rotate(15, np.array([0, 1, 0]))
            elif event.key == pygame.K_z:
                self.selected.rotate(15, np.array([0, 0, 1]))
            elif event.key in [pygame.K_PLUS, pygame.K_EQUALS]:
                self.selected.scale(1.15)
            elif event.key == pygame.K_MINUS:
                self.selected.scale(0.85)
            elif event.key == pygame.K_1:
                self.light.position[0] += 1
            elif event.key == pygame.K_2:
                self.light.position[0] -= 1
            elif event.key == pygame.K_3:
                self.light.position[1] += 1
            elif event.key == pygame.K_4:
                self.light.position[1] -= 1
            elif event.key == pygame.K_u:
                self.selected.diffuse = min(1.0, self.selected.diffuse + 0.1)
            elif event.key == pygame.K_j:
                self.selected.diffuse = max(0.0, self.selected.diffuse - 0.1)
            elif event.key == pygame.K_i:
                self.selected.shininess += 5
            elif event.key == pygame.K_k:
                self.selected.shininess = max(1, self.selected.shininess - 5)


                
    def render(self):
        self.screen.fill((30, 30, 40))
        self.renderer.render_scene(self.screen, self.objects, self.camera, 
                                   self.light, self.shading_mode, self.selected)
        
        # HUD
        texts = [
            f"Objetos: {len(self.objects)}",
            f"Sombreamento: {self.shading_mode.upper()}",
            f"FPS: {int(self.clock.get_fps())}",
        ]
        if self.selected and self.selected in self.objects:
            texts.append(f"Selecionado: Cubo #{self.objects.index(self.selected) + 1}")
        
        for i, text in enumerate(texts):
            surf = self.font.render(text, True, (220, 220, 220))
            self.screen.blit(surf, (10, 10 + i * 25))
        
        pygame.display.flip()
        
    def run(self):
        print("=" * 60)
        print("MODELADOR 3D - Pipeline de Alvy Ray Smith")
        print("=" * 60)
        print("\nControles:")
        print("  WASD/Q/E : Mover objeto")
        print("  X/Y/Z    : Rotacionar")
        print("  +/-      : Escalar")
        print("  N        : Novo cubo")
        print("  TAB      : Próximo objeto")
        print("  DELETE   : Remover")
        print("  F1       : Alternar sombreamento")
        print("  Mouse    : Rotacionar câmera")
        print("  1/2/3/4  : Mover luz")
        print("  U/J      : Aumentar/diminuir difusao") 
        print("  I/K      : Aumentar/diminuir brilho")
        print("  ESC      : Sair")

        print("=" * 60)
        
        while self.running:
            self.handle_events()
            self.render()
            self.clock.tick(60)
        
        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    app = Modeler3D()
    app.run()
