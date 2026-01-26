import time
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
import numpy as np
from std_msgs.msg import Bool 

class PointsGenerator(Node):
    def __init__(self):
        super().__init__('tm5_generatore_punti')
        
        # Canale di pubblicazione dei waypoint
        self.publisher_ = self.create_publisher(Float64MultiArray, 'waypoint', 10)
        
        # Subscriber per il segnale di pronto
        self.ready_to_send = False
        self.create_subscription(Bool, 'robot_ready', self.ready_callback, 10)

    def ready_callback(self, msg):
        if msg.data:
            self.ready_to_send = True

    def send_waypoint(self, points):
        """
        Invia i punti generati assicurandosi che ci sia un ricevitore.
        """
        self.get_logger().info('In attesa del segnale READY dal controller (Unity deve essere in Play)...')

         # Aspetta finché Unity non manda unity_ready = True 
        while rclpy.ok() and not self.ready_to_send: 
            rclpy.spin_once(self, timeout_sec=0.1) 
            time.sleep(0.1) 
            
        # Una volta ricevuto il segnale, invia i punti
        msg = Float64MultiArray()
        msg.data = points.flatten().tolist()
        self.publisher_.publish(msg)
        self.get_logger().info(f'Handshake completato. Inviati {len(points)} punti.')

        # Conferma che l'invio è avvenuto con successo
        return True
    
    def ros_to_unity_position(self, p_ros):
        '''
        Converte le posizioni dal frame di ROS2 al frame di Unity
        ''' 
        p_ros = np.asarray(p_ros) 
        p_unity = np.zeros_like(p_ros) 
        p_unity[:,0] = -p_ros[:,0]  # Unity.x = ROST.x= -ROS.x
        p_unity[:,1] = -p_ros[:,1]  # Unity.z = ROST.y = -ROS.y
        p_unity[:,2] = p_ros[:,2]   # Unity.y = ROST.z = ROS.z
        return p_unity

    def generate_wavy_trajectory_oblique(self, start=np.array([0.3, 0.0, 0.15]), end=np.array([0.9, 0.5, 0.15]),
                                        num_points=100, amplitude=0.02, frequency=3.0):
        """
        Genera una traiettoria obliqua nel piano X-Y con ondulazione sinusoidale.

        """

        # 1. Parametrizzazione lineare della retta
        t = np.linspace(0, 1, num_points)
        line = start + np.outer(t, (end - start))

        # 2. Direzione della retta nel piano XY
        direction = end[:2] - start[:2]
        direction = direction / np.linalg.norm(direction)

        # 3. Vettore ortogonale nel piano XY (per oscillare attorno alla retta)
        normal = np.array([-direction[1], direction[0]])

        # 4. Oscillazione sinusoidale lungo la normale
        wave = amplitude * np.sin(2 * np.pi * frequency * t)

        # 5. Applica l’oscillazione ai punti XY
        line[:, 0] += wave * normal[0]
        line[:, 1] += wave * normal[1]

        return self.ros_to_unity_position(line)
    
    def generate_pocket_square2(self, width=0.12, height=0.18, z=0.15, n_per_side=25, 
                                noise=0.002, offset_x=0.5, offset_y=0.3):
        """
        Genera una traiettoria che simula la tasca di un giubbotto:

        """

        w = width
        h = height
        pts = []

        # --- lato sinistro (dal basso verso l'alto) ---
        for i in range(n_per_side):
            y = i * (h / n_per_side)
            pts.append([0, y, z])

        # --- bordo superiore arcuato ---
        t = np.linspace(0, np.pi, n_per_side)
        for ti in t:
            x = (w/2) * (1 - np.cos(ti))   # arco morbido
            y = h
            pts.append([x, y, z])

        # --- lato destro (dall'alto verso il basso) ---
        for i in range(n_per_side):
            y = h - i * (h / n_per_side)
            pts.append([w, y, z])

        # --- bordo inferiore smussato ---
        t = np.linspace(np.pi, 2*np.pi, n_per_side)
        for ti in t:
            x = (w/2) * (1 - np.cos(ti))
            y = 0
            pts.append([x, y, z])

        pts = np.array(pts)

        # --- rumore ---
        pts[:,0] += noise * np.random.randn(len(pts))
        pts[:,1] += noise * np.random.randn(len(pts))

        # --- offset per allontanare la tasca dall'origine ---
        pts[:,0] += offset_x
        pts[:,1] += offset_y

        return self.ros_to_unity_position(pts)

def main():
    rclpy.init()
    node = PointsGenerator()
    try:
        tasca = node.generate_wavy_trajectory_oblique()
        zip = node.generate_pocket_square2()

        # Costruisci la lista piatta
        flat = [len(tasca), len(zip)] + tasca.flatten().tolist() + zip.flatten().tolist()

        # Invia come array numpy
        success = node.send_waypoint(np.array(flat))

        if success:
            time.sleep(0.5)

    except KeyboardInterrupt:
        node.get_logger().info('Chiusura forzata dall\'utente.')
    except Exception as e:
        print(f"Errore fatale nel main: {e}")
    finally:
        node.destroy_node()


    