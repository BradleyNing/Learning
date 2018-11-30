import java.awt.event.*;
import javax.swing.*;

public class UseInnerClass {
	JFrame frame;

	public UseInnerClass() {
		frame = new JFrame();
		frame.setSize(800, 600);
		frame.setVisible(true);
		frame.addMouseListener(new MouseAdapter(){
			public void mouseClicked(MouseEvent e) {
				frame.setTitle("Cliced position: " + e.getX() + ", " + e.getY());
			}
		});
		frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
	}

	public static void main(String[] args) {
		new UseInnerClass();
	}

}