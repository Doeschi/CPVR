import taichi as ti

ti.init(arch=ti.gpu)

N = 5  # number of quads per direction
NF = 2 * N ** 2  # number of faces
NV = (N + 1) ** 2  # number of vertices
mesh_faces = ti.Vector.field(3, int, NF)  # ids of three vertices of each face

pos = ti.Vector.field(2, float, NV, needs_grad=True)  # position of each vertex
vel = ti.Vector.field(2, float, NV)  # velocity of each vertex
undeformed_state = ti.Matrix.field(2, 2, float, NF)  # captures the shape in undeformed state

dt = 0.00005  # time step
dx = 1 / N  # grid spacing?
density = 40  # density per unit

youngs_modulus = 40_000  # Young's modulus -> stiffness
poissons_ratio = 0.2  # Poisson's ratio -> deformation perpendicular to the force
shearing_modulus = youngs_modulus / 2 / (1 + poissons_ratio)  # shear modulus, resist shape change -> see https://en.wikipedia.org/wiki/Shear_modulus
bulk_modulus = youngs_modulus * poissons_ratio / (1 + poissons_ratio) / (1 - 2 * poissons_ratio)  # resistance to compression -> https://en.wikipedia.org/wiki/Bulk_modulus

ball_pos, ball_radius = ti.Vector([0.5, 0.0]), 0.31
damping = 14.5

deformation_gradient_per_face = ti.Matrix.field(2, 2, float, NF, needs_grad=True)
volume = ti.field(float, NF)
potential_energy_per_face = ti.field(float, NF)  # potential energy of each face (Neo-Hookean)
total_potential_energy = ti.field(float, (), needs_grad=True)  # total potential energy

gravity = ti.Vector.field(2, float, ())
attractor_pos = ti.Vector.field(2, float, ())
attractor_strength = ti.field(float, ())


@ti.kernel
def update_total_potential_energy():
    for i in range(NF):
        ia, ib, ic = mesh_faces[i]

        # get current positions
        a, b, c = pos[ia], pos[ib], pos[ic]

        # calculate the area of the triangle, aka the volume
        volume[i] = abs((a - c).cross(b - c))

        # calculate the deformation
        current_deformation = ti.Matrix.cols([a - c, b - c])
        deformation_gradient_per_face[i] = current_deformation @ undeformed_state[i]

    for i in range(NF):
        deformation_gradient = deformation_gradient_per_face[i]
        # the determinant captures the area change
        log_J_i = ti.log(deformation_gradient.determinant())
        potential_energy = shearing_modulus / 2 * ((deformation_gradient.transpose() @ deformation_gradient).trace() - 2)
        potential_energy -= shearing_modulus * log_J_i
        potential_energy += bulk_modulus / 2 * log_J_i ** 2
        potential_energy_per_face[i] = potential_energy
        total_potential_energy[None] += volume[i] * potential_energy


@ti.kernel
def advance():
    # update velocity
    for i in range(NV):
        acc = -pos.grad[i] / (density * dx ** 2)
        g = gravity[None] * 0.8 + attractor_strength[None] * (attractor_pos[None] - pos[i]).normalized(1e-5)
        vel[i] += dt * (acc + g * 40)
        vel[i] *= ti.exp(-dt * damping)

    for i in range(NV):
        # ball boundary condition:
        distance = pos[i] - ball_pos
        distance_squared = distance.norm_sqr()
        if distance_squared <= ball_radius ** 2:
            NoV = vel[i].dot(distance)
            if NoV < 0:
                vel[i] -= NoV * distance / distance_squared

        cond = (pos[i] < 0) & (vel[i] < 0) | (pos[i] > 1) & (vel[i] > 0)
        # rect boundary condition:
        for j in ti.static(range(pos.n)):
            if cond[j]:
                vel[i][j] = 0
        pos[i] += dt * vel[i]


@ti.kernel
def init_pos():
    # define initial position and velocity
    for x, y in ti.ndrange(N + 1, N + 1):
        vertex_index = x * (N + 1) + y
        size = 0.25
        offset = ti.Vector([0.45, 0.45])
        pos[vertex_index] = ti.Vector([x, y]) / N * size + offset
        vel[vertex_index] = ti.Vector([0, 0])

    for i in range(NF):
        # get the vertex indices of this face
        ia, ib, ic = mesh_faces[i]

        # get the position of the triangle
        a, b, c = pos[ia], pos[ib], pos[ic]

        edge_matrix = ti.Matrix.cols([a - c, b - c])
        undeformed_state[i] = edge_matrix.inverse()


@ti.kernel
def init_mesh():
    # define faces
    for x, y in ti.ndrange(N, N):
        face_index = (x * N + y) * 2
        a = x * (N + 1) + y
        b = a + 1
        c = a + N + 2
        d = a + N + 1
        mesh_faces[face_index + 0] = [a, b, c]
        mesh_faces[face_index + 1] = [c, d, a]


def draw_object(gui):
    pos_ = pos.to_numpy()
    potential_energy_per_face_ = potential_energy_per_face.to_numpy()
    mesh_indices_ = mesh_faces.to_numpy()

    a, b, c = pos_[mesh_indices_[:, 0]], pos_[mesh_indices_[:, 1]], pos_[mesh_indices_[:, 2]]
    k = potential_energy_per_face_ * (10 / youngs_modulus)
    gb = (1 - k) * 0.5
    gui.triangles(a, b, c, color=ti.rgb_to_hex([k + gb, gb, gb]))


def main():
    init_mesh()
    init_pos()

    gravity[None] = [0, -1]

    gui = ti.GUI("FEM128")
    print(
        "[Hint] Use WSAD/arrow keys to control gravity. Use left/right mouse buttons to attract/repel. Press R to reset."
    )
    while gui.running:
        for e in gui.get_events(gui.PRESS):
            if e.key == gui.ESCAPE:
                gui.running = False
            elif e.key == "r":
                init_pos()
            elif e.key in ("a", gui.LEFT):
                gravity[None] = [-1, 0]
            elif e.key in ("d", gui.RIGHT):
                gravity[None] = [+1, 0]
            elif e.key in ("s", gui.DOWN):
                gravity[None] = [0, -1]
            elif e.key in ("w", gui.UP):
                gravity[None] = [0, +1]

        mouse_pos = gui.get_cursor_pos()
        attractor_pos[None] = mouse_pos
        attractor_strength[None] = gui.is_pressed(gui.LMB) - gui.is_pressed(gui.RMB)

        for i in range(50):
            with ti.ad.Tape(loss=total_potential_energy):
                update_total_potential_energy()
            advance()

        draw_object(gui)
        gui.circle(mouse_pos, radius=15, color=0x336699)
        gui.circle(ball_pos, radius=ball_radius * 512, color=0x666666)
        gui.circles(pos.to_numpy(), radius=2, color=0xFFAA33)
        gui.show()


if __name__ == "__main__":
    main()

